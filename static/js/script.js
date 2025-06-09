// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    // Handle file input
    const fileInput = document.getElementById('resumeFile');
    const fileName = document.getElementById('file-name');
    const uploadForm = document.getElementById('resumeForm');
    const uploadStatus = document.getElementById('upload-status');
    
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                fileName.textContent = fileInput.files[0].name;
            } else {
                fileName.textContent = 'No file chosen';
            }
        });
    }
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            if (!fileInput.files.length) {
                uploadStatus.innerHTML = '<span class="status-error">Please select a file first.</span>';
                return;
            }
            
            const formData = new FormData();
            formData.append('resume', fileInput.files[0]);
            
            uploadStatus.innerHTML = '<span>Uploading and analyzing resume...</span>';
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    uploadStatus.innerHTML = `<span class="status-success">${data.message}</span>`;
                    setTimeout(() => {
                        window.location.href = `/chat/${data.chat_id}`;
                    }, 1000);
                } else {
                    uploadStatus.innerHTML = `<span class="status-error">${data.error}</span>`;
                }
            })
            .catch(error => {
                uploadStatus.innerHTML = `<span class="status-error">An error occurred: ${error.message}</span>`;
            });
        });
    }
    
    // Modal handling
    const modal = document.getElementById('renameModal');
    const close = document.getElementsByClassName('close')[0];
    const saveBtn = document.getElementById('saveNewName');
    const newNameInput = document.getElementById('newChatName');
    const chatIdInput = document.getElementById('chatIdToRename');
    
    if (close) {
        close.onclick = function() {
            modal.style.display = "none";
        }
    }
    
    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }
    
    if (saveBtn) {
        saveBtn.onclick = function() {
            const newName = newNameInput.value.trim();
            const chatId = chatIdInput.value;
            
            if (!newName) return;
            
            fetch(`/rename_chat/${chatId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ new_name: newName }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    window.location.reload();
                }
            });
        }
    }
});

function renameChat(chatId) {
    const modal = document.getElementById('renameModal');
    const newNameInput = document.getElementById('newChatName');
    const chatIdInput = document.getElementById('chatIdToRename');
    
    // Find the current chat name
    const chatItems = document.querySelectorAll('.chat-item');
    let currentName = '';
    
    for (const item of chatItems) {
        const link = item.querySelector('.chat-link');
        if (link && link.href.includes(chatId)) {
            currentName = item.querySelector('.chat-name').textContent;
            break;
        }
    }
    
    newNameInput.value = currentName;
    chatIdInput.value = chatId;
    modal.style.display = "block";
    newNameInput.focus();
}
