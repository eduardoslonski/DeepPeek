from app import create_app
import shutil
import os

app = create_app()

if __name__ == '__main__':
    dir_samples_data = "app/samples_data/"
    if os.path.exists(dir_samples_data):
        shutil.rmtree(dir_samples_data)
    os.mkdir(dir_samples_data)
    app.run(debug=True)