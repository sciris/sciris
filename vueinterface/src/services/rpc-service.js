// rpc-service.js -- RPC functions for Vue to call
//
// Last update: 8/30/17 (gchadder3)

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

// readJsonFromBlob(theBlob) -- Attempt to convert a Blob passed in to a JSON.
//   Passes back a Promise.
function readJsonFromBlob (theBlob) {
  return new Promise((resolve, reject) => {
    // Create a FileReader.
    const reader = new FileReader()

    // Create a callback for after the load attempt is finished.
    reader.addEventListener("loadend", function() {
      // reader.result contains the contents of blob as text when this is called

      // Call a resolve passing back a JSON version of this.
      try {
        // Try the conversion.
        var jsonresult = JSON.parse(reader.result)

        // (Assuming successful) make the Promise resolve with the JSON result.
        resolve(jsonresult)
      }
      catch (e) {
        // On failure to convert to JSON, reject the Promise.
        reject(Error('Failed to convert blob to JSON'))
      }
    })

    // Start the load attempt, trying to read the blob in as text.
    reader.readAsText(theBlob)
  })
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
    // Log the download RPC call.
    consoleLogCommand("download", name, args, kwargs)

    // Do the RPC processing, returning results as a Promise.
    return new Promise((resolve, reject) => {
      // Send the POST request for the RPC call.
      axios.post('/api/download', {
        name: name, 
        args: args, 
        kwargs: kwargs
      }, 
      {
        responseType: 'blob'
      })
      .then(response => {
        readJsonFromBlob(response.data)
        .then(responsedata => {
          // If we have error information in the response (which indicates 
          // a logical error on the server side)...
          if (typeof(responsedata.error) != 'undefined') {
            // For now, reject with an error message matching the error.
            reject(Error(responsedata.error))
          }
        })
        .catch(error2 => {
          // An error here indicates we do in fact have a file to download.

          // Create a new blob object (containing the file data) from the
          // response.data component.
          var blob = new Blob([response.data])

          // Grab the file name from response.headers.
          var filename = response.headers.filename

          // Bring up the browser dialog allowing the user to save the file 
          // or cancel doing so.
          filesaver.saveAs(blob, filename)

          // Signal success with the response.
          resolve(response)
        })
      })
      .catch(error => {
        // If there was an actual response returned from the server...
        if (error.response) {
          readJsonFromBlob(error.response.data)
          .then(responsedata => {
            // If we have exception information in the response (which indicates 
            // an exception on the server side)...
            if (typeof(responsedata.exception) != 'undefined') {
              // For now, reject with an error message matching the exception.
              // In the future, we want to put the exception message in a 
              // pop-up dialog.
              reject(Error(responsedata.exception))
            }
          })
          .catch(error2 => {
            // Reject with the error axios got.
            reject(error)
          })

        // Otherwise (no response was delivered), reject with the error axios got.
        } else {
          reject(error)
        }
      })
    })
  },

  rpcUploadCall (name, args, kwargs) {
    console.log('upload RPC')
  }
}