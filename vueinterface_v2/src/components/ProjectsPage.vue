<!-- 
ProjectsPage.vue -- ProjectsPage Vue component

Last update: 1/30/18 (gchadder3)
-->

<template>
  <div class="SitePage">
    <div class="PageSection">
      <h2>Create projects</h2>

      <div>
        Choose a demonstration project from our database:
      </div>

      <select v-model='selectedDemoProject'>
        <option v-for='choice in demoProjectList'>
          {{ choice }}
        </option>
      </select>
      <button @click="addDemoProject">Add this project</button>

      <br>

      <div>
        Or create/upload a new project:
      </div>

      <button @click="createNewProject">Create new project</button>
      <button @click="uploadProjectFromFile">Upload project from file</button>
      <button @click="uploadProjectFromSpreadsheet">Upload project from spreadsheet</button>
    </div>

    <div class="PageSection"
         v-if='projectSummaries.length > 0'>
      <h2>Manage projects</h2>

      <input type="text"
             class="txbox"
             style="margin-bottom: 20px"
             placeholder="Filter Projects"/>

      <table>
        <thead>
          <tr>
            <th>
              <input type="checkbox"/>
            </th>
            <th>Name</th>
            <th>Select</th>
            <th>Created on</th>
            <th>Updated on</th>
            <th>Data uploaded on</th>
            <th>Actions</th>
            <th>Data spreadsheet</th>
            <th>Project file</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for='projectSummary in projectSummaries'>
            <td>
              <input type="checkbox"/>
            </td>
            <td>{{ projectSummary.projectName }}</td>
            <td>
              <button @click="openProject">Open</button>
            </td>
            <td>{{ projectSummary.creationTime }}</td>
            <td>{{ projectSummary.updateTime }}</td>
            <td>{{ projectSummary.spreadsheetUploadTime }}</td>
            <td>
              <button @click="copyProject">Copy</button>
              <button @click="renameProject">Rename</button>
            </td>
            <td>
              <button @click="uploadSpreadsheetToProject">Upload</button>
              <button @click="downloadSpreadsheetFromProject">Download</button>
            </td>
            <td>
              <button @click="downloadProjectFile">Download</button>
              <button @click="downloadProjectFileWithResults">Download with results</button>
            </td>
          </tr>
        </tbody>
      </table>

      <div>
        <button @click="deleteSelectedProjects">Delete selected</button>
        <button @click="downloadSelectedProjects">Download selected</button>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
var filesaver = require('file-saver')
import rpcservice from '../services/rpc-service'
import router from '../router'

export default {
  name: 'ProjectsPage',

  data () {
    return {
      demoProjectList: [],
//      demoProjectList: ['Graph 1', 'Graph 2', 'Graph 3'],
      selectedDemoProject: '',
      projectSummaries: [],
      demoProjectSummaries: 
        [
          {
            projectName: 'Graph 1', 
            creationTime: '2017-Sep-21 08:44 AM',
            updateTime: '2017-Sep-21 08:44 AM',
            spreadsheetUploadTime: '2017-Sep-21 08:44 AM'
          }, 
          {
            projectName: 'Graph 2',
            creationTime: '2017-Sep-22 08:44 AM',
            updateTime: '2017-Sep-22 08:44 AM',
            spreadsheetUploadTime: '2017-Sep-22 08:44 AM'
          }, 
          {
            projectName: 'Graph 3',
            creationTime: '2017-Sep-23 08:44 AM',
            updateTime: '2017-Sep-23 08:44 AM',
            spreadsheetUploadTime: '2017-Sep-23 08:44 AM'
          }
        ],
      selectedgraph: '',
      serverresponse: '',
      loadedfile: '', 
      servererror: '',
      resourcechoices: []
    }
  },

  created () {
    // If we have no user logged in, automatically redirect to the login page.
    if (this.$store.state.currentuser.displayname == undefined) {
      router.push('/login')
    } 

    // Otherwise...
    else {
      //this.demoProjectList = ['Graph 1', 'Graph 2', 'Graph 3']

      // picking from the demoProjectSummaries object
      this.demoProjectList = []
      for (var ii = 0; ii < this.demoProjectSummaries.length; ii++) {
        this.demoProjectList.push(this.demoProjectSummaries[ii].projectName)
      }

      // Initialize the selection of the demo project.
      this.selectedDemoProject = this.demoProjectList[0]

      // Otherwise, get the list of the graphs available.   
      this.updateScatterplotDataList()
    }
  },

  methods: {
    addDemoProject () {
      console.log('addDemoProject() called')

      this.projectSummaries.push(this.demoProjectSummaries[0])
    },

    createNewProject () {
      console.log('createNewProject() called')
    },

    uploadProjectFromFile () {
      console.log('uploadProjectFromFile() called')
    },

    uploadProjectFromSpreadsheet () {
      console.log('uploadProjectFromSpreadsheet() called')
    },

    openProject () {
      console.log('openProject() called')
    },

    copyProject () {
      console.log('copyProject() called')
    },

    renameProject () {
      console.log('renameProject() called')
    },

    uploadSpreadsheetToProject () {
      console.log('uploadSpreadsheetToProject() called')
    },

    downloadSpreadsheetFromProject () {
      console.log('downloadSpreadsheetFromProject() called')
    },

    downloadProjectFile () {
      console.log('downloadProjectFile() called')
    },

    downloadProjectFileWithResults () {
      console.log('downloadProjectFileWithResults() called')
    },

    deleteSelectedProjects () {
      console.log('deleteSelectedProjects() called')
    },

    downloadSelectedProjects () {
      console.log('downloadSelectedProjects() called')
    },

    clearFigureWindow () {
      // If we already have a figure, pop the figure object, and clear
      // the DOM.
      if (mpld3.figures.length > 0) {
        mpld3.figures.pop()
        document.getElementById('fig01').innerHTML = ''
      }
    },

    updateScatterplotDataList () {
      return new Promise((resolve, reject) => {
        rpcservice.rpcCall('list_saved_scatterplotdata_resources')
        .then(response => {
          this.resourcechoices = response.data
          this.selectedgraph = this.resourcechoices[0]
          resolve(response)
        })
        .catch(error => {
          reject(Error(response.data.error))
        })
      })
    },

    sendRequest () {
      // If we already have a figure, pop the figure object, and clear
      // the DOM.
      this.clearFigureWindow()

      // Clear the loaded file.
      this.loadedfile = ''

      // Clear the server error.
      this.servererror = ''

      // Call RPC get_saved_scatterplotdata_graph.
      rpcservice.rpcCall('get_saved_scatterplotdata_graph', [this.selectedgraph])
      .then(response => {
        // Pull out the response data.
        this.serverresponse = response.data

        // Draw the figure in the 'fig01' div tag.
        mpld3.draw_figure('fig01', response.data.graph)

        // Remember the file that was loaded.
        this.loadedfile = this.selectedgraph + '.csv'
      })
      .catch(error => {
        // Pull out the error message.
        this.serverresponse = 'There was an error: ' + error.message

        // Set the server error.
        this.servererror = error.message
      })
    }, 

    downloadFile () {
      // Clear the server error.
      this.servererror = ''

      // Call RPC download_saved_scatterplotdata.
      rpcservice.rpcDownloadCall('download_saved_scatterplotdata', [this.selectedgraph])
      .then(response => {
        // Pull out the response data.
        this.serverresponse = response.data
      })
      .catch(error => {
        // Pull out the error message.
        this.serverresponse = 'There was an error: ' + error.message

        // Set the server error.
        this.servererror = error.message
      })
    },
 
    deleteFile () {
      // Clear the server error.
      this.servererror = ''

      // Call RPC delete_saved_scatterplotdata_graph.
      rpcservice.rpcCall('delete_saved_scatterplotdata_graph', [this.selectedgraph])
      .then(response => {
        // Pull out the response data.
        this.serverresponse = response.data

        // Update the ScatterplotData list.
        this.updateScatterplotDataList().
        then(response => {
          // Clear the loaded file.
          this.loadedfile = ''

          // If we already have a figure, pop the figure object, and clear
          // the DOM.
          this.clearFigureWindow()
        })
      })
      .catch(error => {
        // Pull out the error message.
        this.serverresponse = 'There was an error: ' + error.message

        // Set the server error.
        this.servererror = error.message
      })
    },

    uploadFile () {
      // Clear the server error.
      this.servererror = ''

      // Call RPC upload_scatterplotdata_from_csv.
      rpcservice.rpcUploadCall('upload_scatterplotdata_from_csv', [this.selectedgraph], {}, '.csv')
      .then(response => {
        // Pull out the response data.
        this.serverresponse = response.data

        // Update the ScatterplotData list.
        this.updateScatterplotDataList().
        then(response2 => {
          // Set the selected graph to the new upload.
          this.selectedgraph = response.data

          // Send a request to fetch the new file.
          this.sendRequest()
        })
      })
      .catch(error => {
        // Pull out the error message.
        this.serverresponse = 'There was an error: ' + error.message

        // Set the server error.
        this.servererror = error.message 
      })  
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style lang="scss" scoped>
  $vue-blue: #32485F;
  $vue-green: #00C185;

/*  .PageSection h2 {
    color: red;
  } */
/*  h2 {
    color: $vue-blue;
  } */
</style>
