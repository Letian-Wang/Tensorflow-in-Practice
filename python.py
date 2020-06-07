
from os import getcwd
import os
import shutil
import zipfile

# extract data
path_cats_and_dogs = getcwd() + "/cats-and-dogs.zip"
if not os.path.exists(getcwd() + '/tmp'):
    os.mkdir(getcwd() + '/tmp')
else:
    shutil.rmtree(getcwd() + '/tmp')
local_zip = path_cats_and_dogs
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(getcwd()+'/tmp')
zip_ref.close()
print(len(os.listdir('/tmp/PetImages/Cat/')))
print(len(os.listdir('/tmp/PetImages/Dog/')))

os.path.getsize()
shutil.copyfile(origin_file, destination_file)

# csv
csv_reader = csv.reader(training_file, delimiter=',')
next(csv_reader)
for line in csv_reader:

# numpy format conversion
images.dtype            # check format
images = np.array(images).astype('float')       # convert formatarr2.dtype

# dimension: (10000, 28, 28) -> (10000, 28, 28, 1))
training_images = np.expand_dims(training_images, axis=3)


# JSON
with open(getcwd() + '/sarcasm.json', 'r') as f:
    datastore = json.load(f)
sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

# print
print(val, end=" ") # end with " " not "\n"
