document.addEventListener('DOMContentLoaded', function() {

    const themeToggle = document.getElementById('themeToggle');
    const sunIcon = document.getElementById('sunIcon');
    const moonIcon = document.getElementById('moonIcon');
    const htmlElement = document.documentElement;
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      htmlElement.className = savedTheme;
      updateThemeIcons(savedTheme);
    } else {

      htmlElement.className = 'dark';
      updateThemeIcons('dark');
    }
    
    function updateThemeIcons(theme) {
      if (theme === 'light') {
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'block';
      } else {
        sunIcon.style.display = 'block';
        moonIcon.style.display = 'none';
      }
    }
    
    themeToggle.addEventListener('click', function() {
      if (htmlElement.classList.contains('dark')) {
        htmlElement.className = 'light';
        localStorage.setItem('theme', 'light');
        updateThemeIcons('light');
      } else {
        htmlElement.className = 'dark';
        localStorage.setItem('theme', 'dark');
        updateThemeIcons('dark');
      }
    });

 
    const loginTab = document.getElementById('loginTab');
    const signupTab = document.getElementById('signupTab');
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    const switchToSignup = document.getElementById('switchToSignup');
    const switchToLogin = document.getElementById('switchToLogin');
    
    if (window.location.hash === '#signup') {
      activateSignupTab();
    }

    function activateLoginTab() {
      loginTab.classList.add('active');
      signupTab.classList.remove('active');
      loginForm.classList.add('active');
      signupForm.classList.remove('active');
      window.location.hash = '';
    }
    
    function activateSignupTab() {
      signupTab.classList.add('active');
      loginTab.classList.remove('active');
      signupForm.classList.add('active');
      loginForm.classList.remove('active');
      window.location.hash = 'signup';
    }
    
    loginTab.addEventListener('click', activateLoginTab);
    signupTab.addEventListener('click', activateSignupTab);
    switchToSignup.addEventListener('click', activateSignupTab);
    switchToLogin.addEventListener('click', activateLoginTab);
    
   
    const loginFormElement = document.getElementById('loginForm');
    const signupFormElement = document.getElementById('signupForm');
    
    loginFormElement.addEventListener('submit', async(e) =>{
      e.preventDefault();
      
      const email = document.getElementById('loginEmail').value;
      const password = document.getElementById('loginPassword').value;
      
      if (validateLoginForm(email, password)) {
 
        localStorage.setItem('isLoggedIn', 'true');
        localStorage.setItem('username', email.split('@')[0]);
      
        // if (email.toLowerCase().includes('hr')) {
        //   window.location.href = 'hr-dashboard.html';
        // } else {
        //   window.location.href = 'dashboard-landing.html';
        // }

        }
        const userData={
            email: email,
            password:password
          }
          try {
            const res = await fetch('http://localhost:3000/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(userData)
            });
            } catch (err) {
             console.error('Fetch error:', err);
            }
        
    });
    
    signupFormElement.addEventListener('submit', async(e)=> {
      e.preventDefault();
      
      const name = document.getElementById('signupName').value;
      const email = document.getElementById('signupEmail').value;
      const password = document.getElementById('signupPassword').value;
      const confirmPassword = document.getElementById('signupConfirmPassword').value;
      
      if (validateSignupForm(name, email, password, confirmPassword)) {
    
        localStorage.setItem('isLoggedIn', 'true');
        localStorage.setItem('username', email.split('@')[0]);
        
 
        // if (email.toLowerCase().includes('hr')) {
        //   window.location.href = 'hr-dashboard.html';
        // } else {
        //   window.location.href = 'dashboard-landing.html';
        // }
      }
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
        if (!res.ok) {
            alert(result.error)
            signupFormElement.reset();
            return;
          }
        signupFormElement.reset()
        } catch (err) {
          console.log(err)
        }
        
    });
    
    function validateLoginForm(email, password) {
      let isValid = true;
 
      document.getElementById('loginEmailError').style.display = 'none';
      document.getElementById('loginPasswordError').style.display = 'none';
      
  
      if (!email || !isValidEmail(email)) {
        document.getElementById('loginEmailError').style.display = 'block';
        isValid = false;
      }
      
      
      if (!password) {
        document.getElementById('loginPasswordError').style.display = 'block';
        isValid = false;
      }
      
      return isValid;
    }
    
    function validateSignupForm(name, email, password, confirmPassword) {
      let isValid = true;
      
      document.getElementById('signupNameError').style.display = 'none';
      document.getElementById('signupEmailError').style.display = 'none';
      document.getElementById('signupPasswordError').style.display = 'none';
      document.getElementById('signupConfirmPasswordError').style.display = 'none';
      
      if (!name) {
        document.getElementById('signupNameError').style.display = 'block';
        isValid = false;
      }
      if (!email || !isValidEmail(email)) {
        document.getElementById('signupEmailError').style.display = 'block';
        isValid = false;
      }
      
      if (!password || password.length < 8) {
        document.getElementById('signupPasswordError').style.display = 'block';
        isValid = false;
      }
      if (password !== confirmPassword) {
        document.getElementById('signupConfirmPasswordError').style.display = 'block';
        isValid = false;
      }
      
      return isValid;
    }
    
    function isValidEmail(email) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      return emailRegex.test(email);
    }
  });