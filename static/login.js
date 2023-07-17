document.getElementById("login-form").addEventListener("submit", function(event) {
    event.preventDefault(); // Prevent form submission

    // Get the input values
    var account = document.getElementById("account").value;
    var password = document.getElementById("password").value;

    // Perform validation (you can add your own validation logic here)
    if (account === "" || password === "") {
        alert("Please enter both account and password.");
        return;
    }

    // Login logic (you can replace this with your own login implementation)
    // Here, we assume the login is successful
    alert("Login successful!");

    // Redirect to the index page (you can replace the URL with your actual index page)
    window.location.href = "./index";
});

document.getElementById("forgot-password-link").addEventListener("click", function(event) {
    event.preventDefault(); // Prevent link navigation

    // Add your logic for forgot password functionality here
    alert("Forgot password clicked!");
});

document.getElementById("register-link").addEventListener("click", function(event) {
    event.preventDefault(); // Prevent link navigation

    // Redirect to the register page (you can replace the URL with your actual register page)
    window.location.href = "./register";
});
