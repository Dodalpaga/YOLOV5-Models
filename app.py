from flask import Flask, render_template, request, flash, redirect, Response, send_from_directory
import os,time
import torch
import cv2
import numpy as np
import glob

UPLOAD_FOLDER = './components'
DATASET_FOLDER = './components/Test'
ALLOWED_IMAGES = {'jpg', 'jpeg'}
ALLOWED_LABELS = {'txt'}
app = Flask(__name__)
app.secret_key = "super secret key"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_FOLDER'] = DATASET_FOLDER
app.config['ALLOWED_IMAGES'] = ALLOWED_IMAGES
app.config['ALLOWED_LABELS'] = ALLOWED_LABELS

models_names_from_db = os.listdir("./models")
items = []
for model_names in models_names_from_db:
    items.append(model_names)
sources = {"Laptop Camera":0,"Stand Camera":"http://10.89.155.165:8000/stream.mjpg","Nozzle Camera":"http://10.89.155.77:8000/stream.mjpg","Phone Camera":"http://192.168.93.5:8080/video"}

@app.route('/')
def index():
    return render_template('index.html')

###################################################
########### INTERFACE GALLERY #####################
###################################################
@app.route('/interface', methods=['GET', 'POST'])
def interface():
    models_names_from_db = os.listdir("./models")
    items = []
    for model_name in models_names_from_db:
        items.append(model_name)
    image_names = [f for f in os.listdir('./components/Test/Images_predites') if not f.startswith('.')]
    return render_template("interface.html", image_names=image_names,items=items)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("./components/Test/Images_predites", filename)


###################################################
########### DOCUMENT ##############################
###################################################

@app.route('/doc', methods=['GET'])
def doc():
    return render_template('doc.html')

###################################################
########### UPLOADS ###############################
###################################################

def allowed_images(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGES
           
def allowed_labels(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_LABELS

@app.route('/upload_images', methods=['GET', 'POST'])
def upload_images():
    if request.method == 'POST':
        print("POST")
        # check if the post request has the file part
        if 'upload_images[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        print(request.files)
        for file in request.files.getlist('upload_images[]'):
            print(file)
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                print("No selected file")
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_images(file.filename):
                print("File OK")
                filename = file.filename
                file.save(os.path.join(app.config['DATASET_FOLDER'], filename))
    return redirect("/interface")

@app.route('/upload_labels', methods=['GET', 'POST'])
def upload_labels():
    if request.method == 'POST':
        print("POST")
        # check if the post request has the file part
        if 'upload_labels[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        print(request.files)
        for file in request.files.getlist('upload_labels[]'):
            print(file)
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                print("No selected file")
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_labels(file.filename):
                print("File OK")
                filename = file.filename
                file.save(os.path.join(app.config['DATASET_FOLDER'], filename))
    return redirect("/interface")

###################################################
########### CLEAR CACHE ###########################
###################################################

@app.route('/clear_cache', methods=['GET'])
def clear_cache():
    os.system("rm -rf ./components/Test/Images_predites/*")
    files = glob.glob('./components/Test/*.jpeg') + glob.glob('./components/Test/*.jpg') + glob.glob('./components/Test/*.txt')
    for f in files:
        os.remove(f)
    return redirect("/interface")

###################################################
########### WEBCAM ################################
###################################################

class Detection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, capture_index, model_name):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        print(model_name)
        if model_name:
            model = torch.hub.load('./yolov5', 'custom', path=model_name, force_reload=True, source='local')
            model.conf=0.05
            model.iou=0.45
            print("Model Loaded :!", model_name)
        else:
            model = torch.hub.load('./yolov5', 'yolov5s', pretrained=True, force_reload=True, source='local')
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord, conf = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-2], results.xyxyn[0][:, -2]
        return labels, cord, conf

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, conf_thres):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord, conf = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            conf_i = conf[i]
            if conf_i >= conf_thres:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]) + " " + str(round(float(str(conf_i)[8:-2]),2)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

@app.route("/stream", methods=['POST','GET'])
def stream_page():
    if request.method == 'POST':
        source = request.form['sources']
        if source=='0':
            source=0
        model_name = request.form['model_names']
    else:
        model_name="None"
        source="None"
    print(model_name)
    print(source)
    return render_template("live_streaming.html",items=items,sources=sources,model_name=model_name,source=source)

@app.route('/video_feed', methods=['POST','GET'])
def video_feed():
    def gen(source,model_name):
        if (source!="None") and (model_name!="None"):
            # Par défaut, le script python nous situe au chemin /var/www/html
            detector = Detection(capture_index=source, model_name='./models/{}'.format(model_name)) # capture_index=0 for laptop webcam
            video = detector.get_video_capture()
            width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
            height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
            while True:
                success, image = video.read()
                assert success
                
                frame = cv2.resize(image, (int(width),int(height)))
                
                start_time = time.time()
                results = detector.score_frame(frame)
                frame = detector.plot_boxes(results, frame, conf_thres=detector.model.conf)
                
                end_time = time.time()
                fps = 1/np.round(end_time - start_time, 2)
                    
                cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                
                ret, jpeg = cv2.imencode('.jpg', frame)

                frame = jpeg.tobytes()
                
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    if request.method == 'GET':
        print(request.args)
        source = request.args['source']
        if source=='0':
            source=0
        model_name = request.args['model_name']
        return app.response_class(gen(source,model_name),mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "None"



###################################################
########### INFERENCE #############################
###################################################

def make_inferences_on_folder(model_name="weights.pt",conf_thres=0.25,iou=0.45):

    # Images
    imgs=[]
    imgs_paths = glob.glob("components/Test/*.jpeg") + glob.glob("components/Test/*.jpg")
    for f in imgs_paths:
        img = cv2.imread(f)
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(RGB_img)
        
    if len(imgs)!=0:
        # Model
        model = torch.hub.load("./yolov5","custom",path="./models/"+model_name,source='local')
        model.conf = conf_thres  # confidence threshold (0-1)
        model.iou = iou  # NMS IoU threshold (0-1)
        results = model(imgs)  # custom inference size

        # Results
        results.save(save_dir="components/Test/Images_predites")

@app.route("/inference", methods=['POST'])
def inference():
    if request.method == 'POST':
        conf_thres = request.form['conf_threshold']
        iou = request.form['iou']
        model_name=request.form['model_names']
        make_inferences_on_folder(model_name,conf_thres=float(conf_thres),iou=float(iou))
    return redirect("/interface")


if __name__ == '__main__':
    app.run(debug=True)
    