// rpc-service.js -- RPC functions for Vue to call
//
// Last update: 5/16/18 (gchadder3)

// consoleLogCommand() -- Print an RPC call to the browser console.
function consoleLogCommand (type, funcname, args, kwargs) {
  // Don't show any arguments if none are passed in.
  if (!args) {
    args = ''
  }

  // Don't show any kwargs if none are passed in.
  if (!kwargs) {
    kwargs = ''
  }

  // Print the log line.
  console.log("RPC service call (" + type + "): " + funcname, args, kwargs)
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

// rpcCall() -- normal RPC /api/rpcs calls in scirisapp.py.
function rpcCall (funcname, args, kwargs) {
  // Log the RPC call.
  consoleLogCommand("normal", funcname, args, kwargs)

  // Do the RPC processing, returning results as a Promise.
  return new Promise((resolve, reject) => {
    // Send the POST request for the RPC call.
    axios.post('/api/rpcs', {
      funcname: funcname, 
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
}
  
// rpcDownloadCall() -- download RPC /api/rpcs calls in scirisapp.py.
function rpcDownloadCall (funcname, args, kwargs) {
  // Log the download RPC call.
  consoleLogCommand("download", funcname, args, kwargs)

  // Do the RPC processing, returning results as a Promise.
  return new Promise((resolve, reject) => {
    // Send the POST request for the RPC call.
    axios.post('/api/rpcs', {
      funcname: funcname, 
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
        saveAs(blob, filename)

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
}
  
// rpcUploadCall() -- upload RPC /api/rpcs calls in scirisapp.py.
function rpcUploadCall (funcname, args, kwargs, fileType) {
  // Log the upload RPC call.
  consoleLogCommand("upload", funcname, args, kwargs)

  // Do the RPC processing, returning results as a Promise.
  return new Promise((resolve, reject) => {
    // Function for trapping the change event that has the user-selected 
    // file.
    var onFileChange = (e) => {
      // Pull out the files (should only be 1) that were selected.
      var files = e.target.files || e.dataTransfer.files

      // If no files were selected, reject the promise.
      if (!files.length)
        reject(Error('No file selected'))

      // Create a FormData object for holding the file.
      const formData = new FormData()

      // Put the selected file in the formData object with 'uploadfile' key.
      formData.append('uploadfile', files[0])

      // Add the RPC function name to the form data.
      formData.append('funcname', funcname)

      // Add args and kwargs to the form data.
      formData.append('args', JSON.stringify(args))
      formData.append('kwargs', JSON.stringify(kwargs))

      // Use a POST request to pass along file to the server.
      axios.post('/api/rpcs', formData)
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
    }

    // Create an invisible file input element and set its change callback to 
    // our onFileChange function.
    var inElem = document.createElement('input')
    inElem.setAttribute('type', 'file')
    inElem.setAttribute('accept', fileType)
    inElem.addEventListener('change', onFileChange)

    // Manually click the button to open the file dialog.
    inElem.click()
  })
}