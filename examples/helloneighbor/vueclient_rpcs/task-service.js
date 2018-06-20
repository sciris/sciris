// task-service.js -- task queuing functions for Vue to call
//
// Last update: 6/19/18 (gchadder3)

// sleep() -- sleep for _time_ milliseconds
function sleep(time) {
  return new Promise((resolve) => setTimeout(resolve, time));
}

async function testsleep(interval, reps) {
  for (ii = 0; ii < reps; ii++) {
    console.log('rep #' + ii)
    await sleep(interval * 1000)
  }
}

function getTaskResult(task_id, timeout, func_name, args) {
  return new Promise((resolve, reject) => {
    // Launch the task.
    rpcCall('launch_task', [task_id, func_name, args])
    .then(response => {
      // Sleep timeout seconds.
      sleep(timeout * 1000)
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

// This doesn't work because the while loop just piles up all of the sleep() async calls at once.
function getTaskResult2(task_id, timeout, pollinterval, func_name, args) {
  return new Promise((resolve, reject) => {
    // Initialize a timer to 0 seconds.
    let timer = 0
    
    // Launch the task.
    rpcCall('launch_task', [task_id, func_name, args])
    .then(response => {
      // While we have not gone over the timeout time...
      while (timer < timeout) {
        // Increment the timer.
        timer = timer + pollinterval
        
        // Sleep timeout seconds.
        sleep(pollinterval * 1000)
        .then(response2 => {
          // Check the status of the task.
          rpcCall('check_task', [task_id])
          .then(response3 => {
            // If the task is completed...
            if (response3.data.status == 'completed') {
              // Get the result of the task.
              rpcCall('get_task_result', [task_id])
              .then(response4 => {
                // Signal success with the response.
                resolve(response4)              
              })
              .catch(error => {
                // Reject with the error the task result get attempt gave.
                reject(error)
              })
            }          
          })
        })
      }
      
      // We've timed out, so fail accordingly.
      reject(Error('Task waiting timed out'))
    })
    .catch(error => {
      // Reject with the error the launch gave.
      reject(error)
    })
  }) 
}