// user-service.js -- user functions for Vue to call
//
// Last update: 5/25/18 (gchadder3)

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
function changeUserInfo(username, password, displayname, email) {
  // Get a hex version of a hashed password using the SHA224 algorithm.
  var hashPassword = CryptoApi.hash('sha224', password) 
  
  // Make the actual RPC call.
  return rpcCall('user_change_info', [username, hashPassword, displayname, email])   
}

// changeUserPassword() -- Make an rpcCall() for changing a user's password.
function changeUserPassword(oldpassword, newpassword) {
  // Get a hex version of the hashed passwords using the SHA224 algorithm.
  var hashOldPassword = CryptoApi.hash('sha224', oldpassword)
  var hashNewPassword = CryptoApi.hash('sha224', newpassword)
  
  // Make the actual RPC call.
  return rpcCall('user_change_password', [hashOldPassword, hashNewPassword])   
}

// adminGetUserInfo() -- Make an rpcCall() for getting user information at the admin level.
function adminGetUserInfo(username) {
  // Make the actual RPC call.
  return rpcCall('admin_get_user_info', [username])  
}

// deleteUser() -- Make an rpcCall() for deleting a user.
function deleteUser(username) {
  // Make the actual RPC call.
  return rpcCall('admin_delete_user', [username])   
}

// activateUserAccount() -- Make an rpcCall() for activating a user account.
function activateUserAccount(username) {
  // Make the actual RPC call.
  return rpcCall('admin_activate_account', [username])   
}

// deactivateUserAccount() -- Make an rpcCall() for deactivating a user account.
function deactivateUserAccount(username) {
  // Make the actual RPC call.
  return rpcCall('admin_deactivate_account', [username])   
}

// grantUserAdminRights() -- Make an rpcCall() for granting a user admin rights.
function grantUserAdminRights(username) {
  // Make the actual RPC call.
  return rpcCall('admin_grant_admin', [username])   
}

// revokeUserAdminRights() -- Make an rpcCall() for revoking user admin rights.
function revokeUserAdminRights(username) {
  // Make the actual RPC call.
  return rpcCall('admin_revoke_admin', [username])   
}

// resetUserPassword() -- Make an rpcCall() for resetting a user's password.
function resetUserPassword(username) {
  // Make the actual RPC call.
  return rpcCall('admin_reset_password', [username])   
}