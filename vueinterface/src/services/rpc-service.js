// rpc-service.js -- RPC functions for Vue to call
//
// Last update: 8/29/17 (gchadder3)

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
    // Log the RPC call.
    consoleLogCommand("normal", name, args, kwargs)

    // Do the RPC processing, returning results as a Promise.
    return new Promise((resolve, reject) => {
      // Send the POST request for the RPC call.
      axios.post('/api/procedure', {
        name: name, 
        args: args, 
        kwargs: kwargs
      })
      .then(response => {
        // If there is an error in the POST response.
        if (typeof(response.data.error) != 'undefined') {
          reject(Error(response.data.error))
        }

        // Signal success with the response.
        resolve(response)
      })
      .catch(error => {
        // If there was an actual response returned from the server...
        if (error.response) {
          // If we have exception information in the response (which indicates 
          // an exception on the server side)...
          if (typeof(error.response.data.exception) != 'undefined') {
            // For now, reject with an error message matching the exception.
            // In the future, we want to put the exception message in a 
            // pop-up dialog.
            reject(Error(error.response.data.exception))
          }
        }

        // Reject with the error axios got.
        reject(error)
      })
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