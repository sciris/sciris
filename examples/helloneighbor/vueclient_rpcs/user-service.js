// user-service.js -- user functions for Vue to call
//
// Last update: 5/24/18 (gchadder3)

// loginCall() -- Make an rpcCall() for performing a login.
function loginCall(funcname, username, password) {
  // Get a hex version of a hashed password using the SHA224 algorithm.
  var hashPassword = CryptoApi.hash('sha224', password)
  
  // Make the actual RPC call.
  return rpcCall(funcname, [username, hashPassword])
}

// logoutCall() -- Make an rpcCall() for performing a logout.
function logoutCall(funcname) {
  // Make the actual RPC call.
  return rpcCall(funcname)
}
