const express = require('express');
const path = require('path');
const app = express();

// ✅ Serve static files from your project root
app.use(express.static(path.join(__dirname, 'login_sign_page')));

// Example route (optional)
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'login_sign_page', 'index.html'));
});

const PORT = 3000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));





// Theme toggle functionality
const themeToggle = document.getElementById('themeToggle');
const body = document.body;

// Check if theme preference is stored in localStorage
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'light') {
  body.classList.add('light-theme');
  themeToggle.textContent = '☀';
}

themeToggle.addEventListener('click', () => {
  body.classList.toggle('light-theme');
  
  if (body.classList.contains('light-theme')) {
    localStorage.setItem('theme', 'light');
    themeToggle.textContent = '☀';
  } else {
    localStorage.setItem('theme', 'dark');
    themeToggle.textContent = '🌙';
  }
});

// Tab switching
const loginTab = document.querySelector('[data-tab="login"]');
const signupTab = document.querySelector('[data-tab="signup"]');
const loginForm = document.getElementById('loginForm');
const signupForm = document.getElementById('signupForm');
const showSignup = document.getElementById('showSignup');
const showLogin = document.getElementById('showLogin');

function switchToLogin() {
  loginTab.classList.add('active');
  signupTab.classList.remove('active');
  loginForm.style.display = 'block';
  signupForm.style.display = 'none';
}

function switchToSignup() {
  signupTab.classList.add('active');
  loginTab.classList.remove('active');
  signupForm.style.display = 'block';
  loginForm.style.display = 'none';
}

loginTab.addEventListener('click', switchToLogin);
signupTab.addEventListener('click', switchToSignup);
showSignup.addEventListener('click', (e) => {
  e.preventDefault();
  switchToSignup();
});
showLogin.addEventListener('click', (e) => {
  e.preventDefault();
  switchToLogin();
});

// Form submission handling
loginForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const email = document.getElementById('loginEmail').value.trim();
  const password = document.getElementById('loginPassword').value.trim();
  const loginError = document.getElementById('loginError');
  const loginSuccess = document.getElementById('loginSuccess');
  
  // Simple validation
  if (!email || !password) {
    loginError.textContent = 'Please fill in all fields.';
    loginError.style.display = 'block';
    loginSuccess.style.display = 'none';
    return;
  }

  // Mock authentication (in a real app, this would call an API)
  // For demo, we'll accept any valid-looking email
  if (email.includes('@') && password.length >= 6) {
    loginError.style.display = 'none';
    loginSuccess.textContent = 'Login successful! Redirecting...';
    loginSuccess.style.display = 'block';
    
    // Store in localStorage to simulate authentication
    localStorage.setItem('user', JSON.stringify({
      email: email,
      isLoggedIn: true
    }));
    
    // Redirect after a short delay
    setTimeout(() => {
      window.location.href = './opening_page/index.html';
    }, 1500);
  } else {
    loginError.textContent = 'Invalid email or password.';
    loginError.style.display = 'block';
    loginSuccess.style.display = 'none';
  }


});

signupForm.addEventListener('submit', async(e) => {
  e.preventDefault();
  const name = document.getElementById('signupName').value.trim();
  const email = document.getElementById('signupEmail').value.trim();
  const password = document.getElementById('signupPassword').value.trim();
  const confirmPass = document.getElementById('confirmPassword').value.trim();
  const signupError = document.getElementById('signupError');
  const signupSuccess = document.getElementById('signupSuccess');
  
  // Simple validation
  if (!name || !email || !password || !confirmPass) {
    signupError.textContent = 'Please fill in all fields.';
    signupError.style.display = 'block';
    signupSuccess.style.display = 'none';
    return;
  }

  if (password !== confirmPass) {
    signupError.textContent = 'Passwords do not match.';
    signupError.style.display = 'block';
    signupSuccess.style.display = 'none';
    return;
  }

  if (password.length < 6) {
    signupError.textContent = 'Password must be at least 6 characters.';
    signupError.style.display = 'block';
    signupSuccess.style.display = 'none';
    return;
  }

  // Mock registration (in a real app, this would call an API)
  signupError.style.display = 'none';
  signupSuccess.textContent = 'Account created successfully! You can now log in.';
  signupSuccess.style.display = 'block';
  
  // Store in localStorage to simulate user registration
  localStorage.setItem('registeredUser', JSON.stringify({
    name: name,
    email: email,
    password:password
  }));
  
  const userData={
    name: name,
    email: email,
    password:password
  }
  try {
    const res = await fetch('http://localhost:3000/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userData)
    });

    const result = await res.json();
    console.log(result);
    } catch (err) {
     console.error('Fetch error:', err);
    }


  // Switch to login form after a short delay
  setTimeout(() => {
    switchToLogin();
    document.getElementById('loginEmail').value = email;
  }, 1500);
});

// Check if user is already logged in
const currentUser = JSON.parse(localStorage.getItem('user'));
if (currentUser && currentUser.isLoggedIn) {
  const loginSuccess = document.getElementById('loginSuccess');
  loginSuccess.textContent = 'You are already logged in! Redirecting...';
  loginSuccess.style.display = 'block';
  
  // Redirect after a short delay
  setTimeout(() => {
    window.location.href = './opening_page/index.html';
  }, 1000);
}