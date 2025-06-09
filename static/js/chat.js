// static/js/chat.js
document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    const sendBtn = document.getElementById('send-btn');
    
    // Auto-resize text area based on content
    if (messageInput) {
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }
    
    // Handle sending messages
    if (chatForm) {
        chatForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Disable input while processing
            messageInput.disabled = true;
            sendBtn.disabled = true;
            
            // Add user message to UI
            addMessageToUI('user', message);
            
            // Clear input
            messageInput.value = '';
            messageInput.style.height = 'auto';
            
            // Send message to backend
            fetch('/send_message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    chat_id: currentChatId,
                    message: message
                }),
            })
            .then(response => response.json())
            .then(data => {
                // Add assistant response to UI
                addMessageToUI('assistant', data.response);
                
                // Re-enable input
                messageInput.disabled = false;
                sendBtn.disabled = false;
                messageInput.focus();
            })
            .catch(error => {
                console.error('Error:', error);
                addMessageToUI('assistant', 'An error occurred while processing your request. Please try again.');
                
                // Re-enable input
                messageInput.disabled = false;
                sendBtn.disabled = false;
                messageInput.focus();
            });
        });
    }
    
    // Function to add message to UI
    function addMessageToUI(role, content) {
        const timestamp = new Date().toLocaleString();
        
        // Convert markdown to HTML if it's from assistant
        if (role === 'assistant') {
            // Configure marked to use hljs for code highlighting
            marked.setOptions({
                highlight: function(code, lang) {
                    if (lang && hljs.getLanguage(lang)) {
                        return hljs.highlight(code, { language: lang }).value;
                    }
                    return hljs.highlightAuto(code).value;
                },
                breaks: true
            });
            
            content = marked.parse(content);
        }
        
        const messageHTML = `
            <div class="message-container ${role}">
                <div class="message ${role}">
                    <div class="message-avatar">
                        ${role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>'}
                    </div>
                    <div class="message-content">
                        <div class="message-text">${content}</div>
                        <div class="message-time">${timestamp}</div>
                    </div>
                </div>
            </div>
        `;
        
        // Remove welcome message if it exists
        const welcomeMessage = chatMessages.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        // Add new message
        chatMessages.insertAdjacentHTML('beforeend', messageHTML);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to ask example questions
    window.askExample = function(button) {
        messageInput.value = button.textContent;
        chatForm.dispatchEvent(new Event('submit'));
    };
    
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