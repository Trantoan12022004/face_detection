document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const usersList = document.getElementById('usersList');
    const searchInput = document.getElementById('searchInput');
    const sortSelect = document.getElementById('sortSelect');
    const refreshButton = document.getElementById('refreshButton');
    const userDetailModal = document.getElementById('userDetailModal');
    const confirmDeleteModal = document.getElementById('confirmDeleteModal');
    const closeModal = document.querySelector('.close-modal');
    const userForm = document.getElementById('userForm');
    const deleteUserButton = document.getElementById('deleteUserButton');
    const confirmDeleteButton = document.getElementById('confirmDeleteButton');
    const cancelDeleteButton = document.getElementById('cancelDeleteButton');
    
    // Camera elements
    const startCameraButton = document.getElementById('startCameraButton');
    const captureFaceButton = document.getElementById('captureFaceButton');
    const saveFaceButton = document.getElementById('saveFaceButton');
    const webcamFace = document.getElementById('webcamFace');
    const canvasFace = document.getElementById('canvasFace');
    const capturedFace = document.getElementById('capturedFace');
    
    // Variables
    let users = [];
    let currentUserId = null;
    let capturedImage = null;
    let stream = null;
    
    // Load users on page load
    loadUsers();
    
    // Event listeners
    refreshButton.addEventListener('click', loadUsers);
    searchInput.addEventListener('input', filterUsers);
    sortSelect.addEventListener('change', sortUsers);
    closeModal.addEventListener('click', () => closeUserModal());
    userForm.addEventListener('submit', updateUser);
    deleteUserButton.addEventListener('click', showDeleteConfirmation);
    confirmDeleteButton.addEventListener('click', deleteUser);
    cancelDeleteButton.addEventListener('click', hideDeleteConfirmation);
    
    // Webcam event listeners
    startCameraButton.addEventListener('click', startCamera);
    captureFaceButton.addEventListener('click', captureFace);
    saveFaceButton.addEventListener('click', saveFace);
    
    // Close modal when clicking outside
    window.addEventListener('click', function(event) {
        if (event.target === userDetailModal) {
            closeUserModal();
        }
        if (event.target === confirmDeleteModal) {
            hideDeleteConfirmation();
        }
    });
    
    // Functions
    function loadUsers() {
        usersList.innerHTML = '<div class="loading-indicator">Đang tải danh sách người dùng...</div>';
        
        fetch('http://localhost:5000/api/users')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    users = Object.entries(data.users).map(([id, info]) => ({
                        id: id,
                        ...info
                    }));
                    renderUsers(users);
                } else {
                    usersList.innerHTML = `<div class="error-message">Lỗi: ${data.error}</div>`;
                }
            })
            .catch(error => {
                usersList.innerHTML = `<div class="error-message">Lỗi kết nối: ${error.message}</div>`;
            });
    }
    
    function renderUsers(usersToRender) {
        if (usersToRender.length === 0) {
            usersList.innerHTML = '<div class="no-results">Không có người dùng nào</div>';
            return;
        }
        
        usersList.innerHTML = '';
        
        usersToRender.forEach(user => {
            const faceImagesCount = user.face_images ? user.face_images.length : 
                                    (user.face_image ? 1 : 0);
            
            const card = document.createElement('div');
            card.className = 'user-card';
            card.innerHTML = `
                <h3>${user.name || 'Không có tên'}</h3>
                <div class="user-id">ID: ${user.id}</div>
                <div class="user-info">
                    <div class="user-info-item">
                        <span class="label">Email:</span>
                        <span class="value">${user.email || 'N/A'}</span>
                    </div>
                    <div class="user-info-item">
                        <span class="label">SĐT:</span>
                        <span class="value">${user.phone || 'N/A'}</span>
                    </div>
                    <div class="user-info-item">
                        <span class="label">Ngày tạo:</span>
                        <span class="value">${user.created_at || 'N/A'}</span>
                    </div>
                </div>
                <div class="user-images-count">Số ảnh: ${faceImagesCount}</div>
            `;
            
            card.addEventListener('click', () => openUserDetail(user.id));
            usersList.appendChild(card);
        });
    }
    
    function filterUsers() {
        const searchTerm = searchInput.value.toLowerCase();
        
        const filteredUsers = users.filter(user => {
            return user.name?.toLowerCase().includes(searchTerm) || 
                   user.id.toLowerCase().includes(searchTerm) ||
                   user.email?.toLowerCase().includes(searchTerm) ||
                   user.phone?.toLowerCase().includes(searchTerm) ||
                   user.department?.toLowerCase().includes(searchTerm) ||
                   user.position?.toLowerCase().includes(searchTerm);
        });
        
        renderUsers(filteredUsers);
    }
    
    function sortUsers() {
        const sortBy = sortSelect.value;
        
        const sortedUsers = [...users].sort((a, b) => {
            switch(sortBy) {
                case 'name':
                    return (a.name || '').localeCompare(b.name || '');
                case 'id':
                    return a.id.localeCompare(b.id);
                case 'created_at':
                    const dateA = a.created_at ? new Date(a.created_at) : new Date(0);
                    const dateB = b.created_at ? new Date(b.created_at) : new Date(0);
                    return dateB - dateA; // Newest first
                default:
                    return 0;
            }
        });
        
        renderUsers(sortedUsers);
    }
    
    function openUserDetail(userId) {
        currentUserId = userId;
        
        // Reset form and face images
        userForm.reset();
        document.getElementById('faceImagesContainer').innerHTML = '';
        
        // Reset camera if active
        stopCamera();
        webcamFace.style.display = 'none';
        capturedFace.style.display = 'none';
        captureFaceButton.disabled = true;
        saveFaceButton.disabled = true;
        
        // Show loading in modal
        document.getElementById('modalTitle').textContent = 'Đang tải...';
        userDetailModal.classList.add('show');
        
        // Fetch user details
        fetch(`http://localhost:5000/api/users/${userId}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const user = data.user;
                    
                    // Set modal title
                    document.getElementById('modalTitle').textContent = `Chi tiết: ${user.name}`;
                    
                    // Set form values
                    document.getElementById('userId').value = userId;
                    document.getElementById('userName').value = user.name || '';
                    document.getElementById('userEmail').value = user.email || '';
                    document.getElementById('userPhone').value = user.phone || '';
                    document.getElementById('userPosition').value = user.position || '';
                    document.getElementById('userDepartment').value = user.department || '';
                    
                    // Render face images
                    const faceImagesContainer = document.getElementById('faceImagesContainer');
                    if (user.face_images && user.face_images.length > 0) {
                        user.face_images.forEach((face, index) => {
                            const faceElement = document.createElement('div');
                            faceElement.className = 'face-image-item';
                            faceElement.innerHTML = `
                                <img src="${face.data}" alt="Face ${index + 1}">
                                <div class="face-index">Ảnh ${index + 1}</div>
                                <div class="delete-face" data-index="${index + 1}">×</div>
                            `;
                            faceImagesContainer.appendChild(faceElement);
                            
                            // Add delete event
                            const deleteBtn = faceElement.querySelector('.delete-face');
                            deleteBtn.addEventListener('click', (e) => {
                                e.stopPropagation();
                                deleteFaceImage(userId, parseInt(deleteBtn.dataset.index));
                            });
                        });
                    } else {
                        faceImagesContainer.innerHTML = '<p>Không có ảnh khuôn mặt nào</p>';
                    }
                } else {
                    alert(`Lỗi: ${data.error}`);
                    closeUserModal();
                }
            })
            .catch(error => {
                alert(`Lỗi kết nối: ${error.message}`);
                closeUserModal();
            });
    }
    
    function closeUserModal() {
        userDetailModal.classList.remove('show');
        stopCamera();
    }
    
    function updateUser(event) {
        event.preventDefault();
        
        const userId = document.getElementById('userId').value;
        const userData = {
            name: document.getElementById('userName').value,
            email: document.getElementById('userEmail').value,
            phone: document.getElementById('userPhone').value,
            position: document.getElementById('userPosition').value,
            department: document.getElementById('userDepartment').value
        };
        
        fetch(`http://localhost:5000/api/users/${userId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(userData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Cập nhật thông tin thành công');
                loadUsers();
            } else {
                alert(`Lỗi: ${data.error}`);
            }
        })
        .catch(error => {
            alert(`Lỗi kết nối: ${error.message}`);
        });
    }
    
    function showDeleteConfirmation() {
        confirmDeleteModal.classList.add('show');
    }
    
    function hideDeleteConfirmation() {
        confirmDeleteModal.classList.remove('show');
    }
    
    function deleteUser() {
        const userId = document.getElementById('userId').value;
        
        fetch(`http://localhost:5000/api/users/${userId}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            hideDeleteConfirmation();
            closeUserModal();
            
            if (data.success) {
                alert(data.message);
                loadUsers();
            } else {
                alert(`Lỗi: ${data.error}`);
            }
        })
        .catch(error => {
            hideDeleteConfirmation();
            alert(`Lỗi kết nối: ${error.message}`);
        });
    }
    
    function startCamera() {
        if (stream) {
            stopCamera();
            return;
        }
        
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(videoStream) {
                stream = videoStream;
                webcamFace.srcObject = stream;
                webcamFace.style.display = 'block';
                capturedFace.style.display = 'none';
                startCameraButton.textContent = 'Tắt camera';
                captureFaceButton.disabled = false;
                saveFaceButton.disabled = true;
            })
            .catch(function(error) {
                alert(`Không thể truy cập camera: ${error.message}`);
            });
    }
    
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            webcamFace.srcObject = null;
            startCameraButton.textContent = 'Bật camera';
        }
    }
    
    function captureFace() {
        if (!stream) return;
        
        const context = canvasFace.getContext('2d');
        canvasFace.width = webcamFace.videoWidth;
        canvasFace.height = webcamFace.videoHeight;
        context.drawImage(webcamFace, 0, 0, canvasFace.width, canvasFace.height);
        
        capturedImage = canvasFace.toDataURL('image/jpeg');
        capturedFace.src = capturedImage;
        capturedFace.style.display = 'block';
        webcamFace.style.display = 'none';
        saveFaceButton.disabled = false;
    }
    
    function saveFace() {
        if (!capturedImage || !currentUserId) return;
        
        fetch(`http://localhost:5000/api/users/${currentUserId}/faces`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: capturedImage
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert(data.message);
                // Refresh user details to show new face
                openUserDetail(currentUserId);
            } else {
                alert(`Lỗi: ${data.error}`);
            }
        })
        .catch(error => {
            alert(`Lỗi kết nối: ${error.message}`);
        });
    }
    
    function deleteFaceImage(userId, faceIndex) {
        if (!confirm(`Bạn có chắc chắn muốn xóa ảnh khuôn mặt số ${faceIndex}?`)) {
            return;
        }
        
        fetch(`http://localhost:5000/api/users/${userId}/faces/${faceIndex}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert(data.message);
                // Refresh user details
                openUserDetail(userId);
            } else {
                alert(`Lỗi: ${data.error}`);
            }
        })
        .catch(error => {
            alert(`Lỗi kết nối: ${error.message}`);
        });
    }
});