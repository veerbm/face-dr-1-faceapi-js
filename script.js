// Storing the HTML video element to video

const video = document.getElementById('video')


// Initializing the faceMatcher and labledFaceDescriptors that will be used in to store embeddings and an object of faceMatcher respectively

let labeledFaceDescriptors;
let faceMatcher;


//Loading all the models and once the models are loaded the run the startVideo function 
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceExpressionNet.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(startVideo)


//This funtion start the webstream and load the descriptors of the reference images provided and load a faceMatcher object
//It is made asynchronous as loading the descriptors of the refernce image provided may take some time

async function startVideo() {
  //access the webcam of the device
  navigator.getUserMedia(
    { video: {} },
    stream => video.srcObject = stream,
    err => console.error(err)
  )

  //loadLabelImages funtion defined returns a labeledFaceDescriptor object that holds the descriptor of the reference image
  //along with the label
  
  labeledFaceDescriptors = await loadLabeledImages()
  
  //faceMatcher is used to get the labels of the reference images with a distance less than 0.7
  
  faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.7)
  }

//function to extract the descriptors(embedding) of the refernce images and return LabeledFaceDescriptors object
function loadLabeledImages() {
  const labels = ['Andrew Ng','Steve Jobs','Scarlett Johansson', 'Veer', 'Aman']
  return Promise.all(
    labels.map(async label => {
      const descriptions = []
      //we will iterate over two images of each
      for (let i = 1; i <= 4; i++) {
        //converting the images to HTML image element using fetchImage and extracting the landmarks and descriptor
        const img = await faceapi.fetchImage(`/Images/${label}/${i}.jpg`)
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
        //appending the extracted information to an array
        descriptions.push(detections.descriptor)
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions)
    })
  )
}


  //Here we add the eventListener to the video and perform the function when the video plays
  
  video.addEventListener('play', () => {
    //We create a canvas that holds the current frame of the video
    const canvas = faceapi.createCanvasFromMedia(video)
    //append that canvas to body
    document.body.append(canvas)
    console.log('canvas created', canvas)
    //displaySize will be used the align the dimension of the canvas with the video element
    const displaySize = { width: video.width, height: video.height }
    //mactching the dimension of the canvas to the dimensions mentioned in the displaySize
    faceapi.matchDimensions(canvas, displaySize)
    //Now as the canvas is ready now we will be detecting and recognising the face at an interval of 100ms
    setInterval(async () => {
      //detection will hold the coordinated of the bounding box, facial landmarks, and the descriptor of the image detected in the current frame of video
      const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors()
      console.log('detections', detections)
      //We will resize the detection to match the dimension of the canvas
      const resizedDetections = faceapi.resizeResults(detections, displaySize)
      console.log(resizedDetections)
      //clear the bounding box and labels of the previous frame to prevent overlapping
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
      //result will hold the best match image along with the confidence score of all the images detected in the frame
      const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
      //Iterating over all the detected images
       results.forEach((result, i) => {
      //box will hold the coordinates and other information of the bounding box
      const box = resizedDetections[i].detection.box
      console.log(box)
      //draw the bounding box with required configuration and display the labels
      const drawBox = new faceapi.draw.DrawBox(box, { label: result.label.toString().toUpperCase(), lineWidth:6, boxColor: "rgba(51, 255, 255, 1)", 
                                                    drawLabelOptions:{ fontColor:"rgba(0, 255, 0,1)", fontSize: 40, fontStyle:"Calibri",
                                                    anchorPosition:"BOTTOM_LEFT", backgroundColor:"rgba(0,0,0,0)"}})
      drawBox.draw(canvas)
    })
    }, 100)
  })





