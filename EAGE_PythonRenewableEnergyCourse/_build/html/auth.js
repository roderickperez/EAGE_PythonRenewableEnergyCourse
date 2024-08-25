var allowedUsers = {
    "roderickperez": "123456",
    "eage": "pythonrenewables"
};

var username = prompt("Enter your username:");
var password = prompt("Enter your password:");

if (allowedUsers[username] === password) {
    alert("Access Granted. Welcome, " + username + "!");
} else {
    document.body.innerHTML = "<h1>Access Denied</h1>";
}