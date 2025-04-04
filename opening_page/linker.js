const submit = document.getElementsByClassName("submit")
const email = document.getElementsByClassName("email")
const username = document.getElementsByClassName("username")
const password = document.getElementsByClassName("password")


submit[0].addEventListener("click", async () => {
    const userData = {
        name: username[0].value.trim(),
        email: email[0].value.trim(),
        password: password[0].value.trim()
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
})

