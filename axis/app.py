from flask import Flask, request

app = Flask(__name__)

@app.route("/api/licenseplateverifier", methods=["POST"])
def post_license_plate_event():
  """
  Posts a license plate event to the server.

  Args:
    event_type: The type of event.
    license_plate_number: The license plate number.
    image: The image of the license plate.

  Returns:
    The response from the server.
  """
  data = request.json
  event_type = request.json["event_type"]
  license_plate_number = request.json["license_plate_number"]
  image = request.json["image"]
  print(data)

  # Do something with the event.
  # For example, you could store the license plate number in a database, send an alert, or open a door.

  response = {
    "status": "success"
  }

  return response, 200

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8080)
