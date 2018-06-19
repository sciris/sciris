// task-service.js -- task queuing functions for Vue to call
//
// Last update: 6/19/18 (gchadder3)

// sleep() -- sleep for _time_ milliseconds
function sleep(time) {
  return new Promise((resolve) => setTimeout(resolve, time));
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