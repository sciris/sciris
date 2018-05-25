// user-service.js -- user functions for Vue to call
//
// Last update: 5/24/18 (gchadder3)

// loginCall() -- Make an rpcCall() for performing a login.
function loginCall(username, password) {
  // Get a hex version of a hashed password using the SHA224 algorithm.
  var hashPassword = CryptoApi.hash('sha224', password)
  
  // Make the actual RPC call.
  return rpcCall('user_login', [username, hashPassword])
}

// logoutCall() -- Make an rpcCall() for performing a logout.
function logoutCall() {
  // Make the actual RPC call.
  return rpcCall('user_logout')
}

// getCurrentUserInfo() -- Make an rpcCall() for reading the currently
// logged in user.
function getCurrentUserInfo() {
  // Make the actual RPC call.
  return rpcCall('get_current_user_info')  
}

// getAllUsersInfo() -- Make an rpcCall() for reading all of the users.
function getAllUsersInfo() {
  // Make the actual RPC call.
  return rpcCall('get_all_users')   
}

// registerUser() -- Make an rpcCall() for registering a new user.
function registerUser(username, password, displayname, email) {
  // Get a hex version of a hashed password using the SHA224 algorithm.
  var hashPassword = CryptoApi.hash('sha224', password)

  // Make the actual RPC call.
  return rpcCall('user_register', [username, hashPassword, displayname, email]) 
}

// changeUserInfo() -- Make an rpcCall() for changing a user's info.
/* function changeUserInfo() {
} */

// deleteUser() -- Make an rpcCall() for deleting a user.
function deleteUser(username) {
  // Make the actual RPC call.
  return rpcCall('admin_delete_user', [username])   
}