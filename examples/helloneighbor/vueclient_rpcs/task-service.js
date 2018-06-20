// task-service.js -- task queuing functions for Vue to call
//
// Last update: 6/20/18 (gchadder3)

// sleep() -- sleep for _time_ milliseconds
function sleep(time) {
  // Return a promise that resolves after _time_ milliseconds.
  return new Promise((resolve) => setTimeout(resolve, time));
}

// getTaskResultWaiting() -- given a task_id string, a waiting time (in 
// sec.), and a remote task function name and its args, try to launch 
// the task, then wait for the waiting time, then try to get the 
// result.
function getTaskResultWaiting(task_id, waitingtime, func_name, args) {
  return new Promise((resolve, reject) => {
    // Launch the task.
    rpcCall('launch_task', [task_id, func_name, args])
    .then(response => {
      // Sleep waitingtime seconds.
      sleep(waitingtime * 1000)
      .then(response2 => {
        // Get the result of the task.
        rpcCall('get_task_result', [task_id])
        .then(response3 => {
          // Signal success with the response.
          resolve(response3)              
        })
        .catch(error => {
          // Reject with the error the task result get attempt gave.
          reject(error)
        })         
      })     
    })
    .catch(error => {
      // Reject with the error the launch gave.
      reject(error)
    })
  }) 
}

// getTaskResultPolling() -- given a task_id string, a timeout time (in 
// sec.), a polling interval (also in sec.), and a remote task function name
//  and its args, try to launch the task, then start the polling if this is 
// successful, returning the ultimate results of the polling process. 
function getTaskResultPolling(task_id, timeout, pollinterval, func_name, args) {
  return new Promise((resolve, reject) => {
    // Launch the task.
    rpcCall('launch_task', [task_id, func_name, args])
    .then(response => {
      // Do the whole sequence of polling steps, starting with the first 
      // (recursive) call.
      pollStep(task_id, timeout, pollinterval, 0)
      .then(response2 => {
        // Resolve with the final polling result.
        resolve(response2)
      })
      .catch(error => {
        // Reject with the error the polling gave.
        reject(error)
      })      
    })
    .catch(error => {
      // Reject with the error the launch gave.
      reject(error)
    })
  }) 
} 

// pollStep() -- A polling step for getTaskResultPolling().  Uses the task_id, 
// a timeout value (in sec.) a poll interval (in sec.) and the time elapsed 
// since the start of the entire polling process.  If timeout is zero or 
// negative, no timeout check is applied.  Otherwise, an error will be 
// returned if the polling has gone on beyond the timeout period.  Otherwise, 
// this function does a sleep() and then a check_task().  If the task is 
// completed, it will get the result.  Otherwise, it will recursively spawn 
// another pollStep().
function pollStep(task_id, timeout, pollinterval, elapsedtime) {
  return new Promise((resolve, reject) => {
    // Check to see if the elapsed time is longer than the timeout (and we 
    // have a timeout we actually want to check against) and if so, 
    // fail.
    if ((elapsedtime > timeout) && (timeout > 0)) {
      reject(Error('Task polling timed out'))
    }
    
    // Otherwise, we've not run out of time yet, so do a polling step.
    else {
      // Sleep timeout seconds.
      sleep(pollinterval * 1000)
      .then(response => {
        // Check the status of the task.
        rpcCall('check_task', [task_id])
        .then(response2 => {
          // If the task is completed...
          if (response2.data.task.status == 'completed') {
            // Get the result of the task.
            rpcCall('get_task_result', [task_id])
            .then(response3 => {
              // Signal success with the response.
              resolve(response3)             
            })
            .catch(error => {
              // Reject with the error the task result get attempt gave.
              reject(error)
            })
          }   

          // Otherwise, do another poll step, passing in an incremented 
          // elapsed time.
          else {
            pollStep(task_id, timeout, pollinterval, elapsedtime + pollinterval)
            .then(response3 => {
              // Resolve with the result of the next polling step (which may 
              // include subsequent (recursive) steps.
              resolve(response3)
            })
          }          
        })
      }) 
    }    
  }) 
}