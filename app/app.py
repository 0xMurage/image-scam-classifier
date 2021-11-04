import json, io
from uuid import uuid4
from flask import Flask, request, jsonify, g
from PIL import Image
from predict import initialize, predict_image, predict_url

app = Flask(__name__)

# 30MB Max image size limit
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024

network_input_size, data_labels = initialize()


@app.post('/api/v1/classify/image')
def file_handler():
    """"
    Body:
         file(octet-stream: a single image file
         imageData(multipart/form-data) multiple file images
    Returns:
        results(dict): A json encoded prediction results.
    """
    try:
        if ('imageData' in request.files):
            image_data = request.files['imageData']
        elif ('imageData' in request.form):
            image_data = request.form['imageData']
        else:
            image_data = io.BytesIO(request.get_data())

        if image_data is None:
            return [{'error': 'Image file not received by the server'}], 400

        img = Image.open(image_data)
        results = predict_image(img, network_input_size, data_labels)
        return jsonify({'results': results, 'iteration': 'v1.0.0', 'ref': uuid4()})
    except Exception as e:
        print('EXCEPTION:', str(e))
        return 'Error processing image', 500


@app.post('/api/v1/classify/url')
def url_handler():
    """"
    Body:
         url(string): uri to an image file
    Returns:
        results(dict): A json encoded prediction results.
    """
    try:
        image_url = json.loads(request.get_data().decode('utf-8'))['url']
        results = predict_url(image_url, network_input_size, data_labels)
        return jsonify({'data': results, 'ref': uuid4()})
    except Exception as e:
        print('EXCEPTION:', str(e))
        return {'error': 'Error processing image'}


@app.get('/api/v1/labels')
def labels():
    return {'data': [{l: l} for l in data_labels], 'app_version': 'v2'}


@app.post('/api/v1/feedback/<uuid:ref>')
def user_feedback(ref):
    # save it into db
    return {'message': 'Thank you for the feedback', 'ref': ref}


@app.errorhandler(400)
def not_found(error):
    return {'error': 'Invalid request'}, 400


@app.errorhandler(404)
def not_found(error):
    return {'error': 'Not found. This is awkward but we can\'t find what you are looking for.'}, 404


@app.errorhandler(405)
def illegal_method(error):
    return {'error': 'Method not supported'}, 405


@app.errorhandler(500)
@app.errorhandler(Exception)
def general_error(error):
    print(error)
    return {'error': 'Error encountered while processing your request'}, 500
