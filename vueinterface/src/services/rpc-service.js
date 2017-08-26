// rpc-service.js -- RPC functions for Vue to call
//
// Last update: 8/24/17 (gchadder3)

import axios from 'axios'
var filesaver = require('file-saver')

// consoleLogCommand() -- Print an RPC call to the browser console.
function consoleLogCommand (type, name, args, kwargs) {
  // Don't show any arguments if none are passed in.
  if (!args) {
    args = ''
  }

  // Don't show any kwargs if none are passed in.
  if (!kwargs) {
    kwargs = ''
  }

  // Print the log line.
  console.log("RPC service call (" + type + "): " + name, args, kwargs)
}

export default {
  rpcCall (name, args, kwargs) {
    consoleLogCommand("normal", name, args, kwargs)
    return axios.post('/api/procedure', {
      name: name, 
      args: args, 
      kwargs: kwargs
    })
  },

  rpcDownloadCall (name, args, kwargs) {
    consoleLogCommand("normal", name, args, kwargs)
    return axios.post('/api/download', {
      name: name, 
      args: args, 
      kwargs: kwargs
    }, 
    {
      responseType: 'blob'
    })
  },

  rpcUploadCall (name, args, kwargs) {
    console.log('upload RPC')
  }
}