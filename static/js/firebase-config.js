// static/js/firebase-config.js

// Firebase SDK configuration
const firebaseConfig = {
    apiKey: "AIzaSyAzE5zX6OSOWQg2vTZ8QgizsCKvStRoy_0",
    authDomain: "nlp-project-d88b9.firebaseapp.com",
    projectId: "nlp-project-d88b9",
    storageBucket: "nlp-project-d88b9.firebasestorage.app",
    messagingSenderId: "268333597875",
    appId: "1:268333597875:web:9b55011e8b201bcadbdcc6",
    measurementId: "G-BL1L4WXHHX"
  };
  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);
  
  // Get Firestore instance
  const db = firebase.firestore();
  const storage = firebase.storage();
  
  // Function to fetch all chats for the current user
  async function fetchChats() {
    try {
      const response = await fetch('/api/chats');
      const chats = await response.json();
      return chats;
    } catch (error) {
      console.error('Error fetching chats:', error);
      return [];
    }
  }
  
  // Function to create a new chat
  async function createChat(title, type) {
    try {
      const response = await fetch('/api/chats', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title, type }),
      });
      
      return await response.json();
    } catch (error) {
      console.error('Error creating chat:', error);
      return null;
    }
  }
  
  // Function to fetch a specific chat with messages
  async function fetchChat(chatId) {
    try {
      const response = await fetch(`/api/chats/${chatId}`);
      const chat = await response.json();
      return chat;
    } catch (error) {
      console.error('Error fetching chat:', error);
      return null;
    }
  }
  
  // Function to send a message
  async function sendMessage(chatId, text, sender = 'user') {
    try {
      const response = await fetch(`/api/chats/${chatId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text, sender }),
      });
      
      return await response.json();
    } catch (error) {
      console.error('Error sending message:', error);
      return null;
    }
  }
  
  // Function to clear chat messages
  async function clearChatMessages(chatId) {
    try {
      const response = await fetch(`/api/chats/${chatId}/messages`, {
        method: 'DELETE',
      });
      
      return await response.json();
    } catch (error) {
      console.error('Error clearing messages:', error);
      return null;
    }
  }
  
  // Function to delete a chat
  async function deleteChat(chatId) {
    try {
      const response = await fetch(`/api/chats/${chatId}`, {
        method: 'DELETE',
      });
      
      return await response.json();
    } catch (error) {
      console.error('Error deleting chat:', error);
      return null;
    }
  }
  
  // Function to upload a resume
  async function uploadResume(chatId, file) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`/api/upload-resume/${chatId}`, {
        method: 'POST',
        body: formData,
      });
      
      return await response.json();
    } catch (error) {
      console.error('Error uploading resume:', error);
      return null;
    }
  }