Muhamad Syahid Burhanudien Robbani

120450092

RC

Three Ways of Storing and Accessing Lots of Images in Python

https://realpython.com/storing-images-in-python/

##**Setup**

Dataset gambar yang digunakan: https://www.cs.toronto.edu/~kriz/cifar.html

Menggunakan Versi Python

Dataset yang akan digunakan adalah dataset gambar Canadian Institute for Advanced Research, yang lebih dikenal sebagai CIFAR-10, yang terdiri dari 60.000 gambar berwarna berukuran 32x32 piksel yang termasuk ke dalam berbagai kelas objek, seperti anjing, kucing, dan pesawat terbang. Secara relatif, CIFAR bukanlah dataset yang sangat besar, tetapi jika kita menggunakan dataset TinyImages lengkap, maka Anda akan memerlukan sekitar 400GB ruang disk yang tersedia, yang mungkin akan menjadi faktor pembatas.


Pickle Python memiliki keunggulan kunci dalam kemampuannya untuk meng-serialize objek Python tanpa memerlukan kode atau transformasi tambahan dari kita. Ini membuatnya menjadi alat yang sangat berguna untuk menyimpan dan memuat ulang objek Python dengan mudah dan tanpa kerumitan tambahan. Sebagai peneliti, fitur ini memungkinkan kita untuk dengan cepat menyimpan dan memanipulasi data secara efisien dalam lingkungan Python.

Kode berikut membuka setiap file batch dan memuat semua gambar ke dalam array NumPy:


```python
import numpy as np
import pickle
from pathlib import Path

# Path to the unzipped CIFAR data
data_dir = Path("/content/drive/MyDrive/cifar-10-batches-py")

# Unpickle function provided by the CIFAR hosts
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

images, labels = [], []
for batch in data_dir.glob("data_batch_*"):
    batch_data = unpickle(batch)
    for i, flat_im in enumerate(batch_data[b"data"]):
        im_channels = []
        # Each image is flattened, with channels in order of R, G, B
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024 : (j + 1) * 1024].reshape((32, 32))
            )
        # Reconstruct the original image
        images.append(np.dstack((im_channels)))
        # Save the label
        labels.append(batch_data[b"labels"][i])

print("Loaded CIFAR-10 training set:")
print(f" - np.shape(images)     {np.shape(images)}")
print(f" - np.shape(labels)     {np.shape(labels)}")
```

    Loaded CIFAR-10 training set:
     - np.shape(images)     (50000, 32, 32, 3)
     - np.shape(labels)     (50000,)


Download Library yang dibutuhkan : Pillow, hdf5, dan lmdb

**Pillow**: Manipulasi Gambar

**hdf5**: HDF5 merupakan singkatan dari Hierarchical Data Format, sebuah format file yang disebut sebagai HDF4 atau HDF5. Penelitian menemukan bahwa HDF5 adalah versi yang saat ini dipelihara, dan memiliki asal-usulnya di National Center for Supercomputing Applications sebagai format data ilmiah yang portabel dan ringkas. Dalam penelitian ini, diketahui bahwa HDF5 digunakan secara luas, bahkan oleh lembaga-lembaga terkemuka seperti NASA untuk proyek Earth Data mereka. File HDF terdiri dari dua jenis objek: Dataset dan Grup. Dataset adalah larik multidimensi, sementara grup terdiri dari dataset atau grup lainnya. Diketahui bahwa meskipun dataset harus berisi larik N-dimensi yang homogen, pengguna masih dapat mencapai heterogenitas yang diperlukan dengan menggunakan grup dan dataset yang bersarang.

**lmdb**: LMDB, kadang-kadang disebut sebagai "Lightning Database," merupakan singkatan dari Lightning Memory-Mapped Database karena kecepatannya dan penggunaan file yang dipetakan ke memori. Ini merupakan penyimpanan kunci-nilai, bukan basis data relasional. Dalam hal implementasi, LMDB adalah pohon B+, yang pada dasarnya merupakan struktur grafik mirip pohon yang disimpan di memori di mana setiap elemen kunci-nilai adalah sebuah node, dan node dapat memiliki banyak anak. Kritis, komponen kunci dari pohon B+ diatur untuk sesuai dengan ukuran halaman dari sistem operasi host, memaksimalkan efisiensi saat mengakses pasangan kunci-nilai dalam basis data. Selain itu, efisiensi LMDB juga disebabkan karena peta memori. Ini berarti bahwa LMDB mengembalikan pointer langsung ke alamat memori dari kunci dan nilai, tanpa perlu menyalin apa pun di memori seperti kebanyakan basis data lainnya.


```python
pip install Pillow
```

    Requirement already satisfied: Pillow in /usr/local/lib/python3.10/dist-packages (9.4.0)



```python
pip install hdf5
```

    [31mERROR: Could not find a version that satisfies the requirement hdf5 (from versions: none)[0m[31m
    [0m[31mERROR: No matching distribution found for hdf5[0m[31m
    [0m


```python
pip install lmdb
```

    Collecting lmdb
      Downloading lmdb-1.4.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (299 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m299.2/299.2 kB[0m [31m5.6 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: lmdb
    Successfully installed lmdb-1.4.1


##**Storing a Single Image**

Sekarang setelah kita punya gambaran tentang metode yang akan dibandingkan, mari langsung lihat perbandingan berapa lama waktu yang dibutuhkan untuk membaca dan menulis file, serta berapa banyak memori disk yang akan digunakan. Ini juga akan menjadi pengantar dasar tentang cara kerja metode-metode tersebut, dengan contoh kode penggunaannya. Penting untuk membedakan, karena beberapa metode mungkin dioptimalkan untuk operasi dan jumlah file yang berbeda. Untuk eksperimen, kita dapat membandingkan performa di berbagai jumlah file, dari satu gambar hingga 100.000 gambar. Dengan menggunakan lima batch CIFAR-10 yang berisi total 50.000 gambar, setiap gambar dapat digunakan dua kali untuk mencapai jumlah 100.000 gambar. Untuk mempersiapkan eksperimen, kita perlu membuat folder untuk setiap metode, yang akan berisi semua file basis data atau gambar, dan menyimpan jalur ke direktori-direktori tersebut dalam variabel.

Untuk mempersiapkan eksperimen, perlu dibuat folder untuk setiap metode, yang akan berisi semua file basis data atau gambar, dan menyimpan jalur ke direktori-direktori tersebut dalam variabel.


```python
from pathlib import Path

disk_dir = Path("data/disk/")
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5/")
```

Pastikan folder berhasil dibuat


```python
disk_dir.mkdir(parents=True, exist_ok=True)
lmdb_dir.mkdir(parents=True, exist_ok=True)
hdf5_dir.mkdir(parents=True, exist_ok=True)
```

**Storing to Disk**

Input untuk eksperimen ini adalah sebuah gambar tunggal, yang saat ini ada di memori sebagai array NumPy. Gambar tersebut akan disimpan terlebih dahulu ke disk sebagai gambar .png, dan diberi nama menggunakan ID gambar yang unik, image_id. Ini dapat dilakukan menggunakan paket Pillow yang telah diinstal sebelumnya.


```python
from PIL import Image
import csv

def store_single_disk(image, image_id, label):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")

    with open(disk_dir / f"{image_id}.csv", "wt") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([label])
```

Dalam semua aplikasi nyata, meta data yang terlampir pada gambar juga penting, yang dalam dataset contoh kita adalah label gambar. Saat menyimpan gambar ke disk, ada beberapa opsi untuk menyimpan meta data. Salah satu solusinya adalah menyandikan label ke dalam nama gambar. Ini memiliki keuntungan tidak memerlukan file tambahan, tetapi juga memiliki kelemahan besar karena memaksa Anda untuk berurusan dengan semua file saat melakukan apapun dengan label. Menyimpan label dalam file terpisah memungkinkan Anda untuk bermain-main dengan label sendiri, tanpa harus memuat gambar. Di atas, label disimpan dalam file .csv terpisah untuk eksperimen ini. Sekarang mari lanjutkan dengan melakukan tugas yang sama dengan LMDB.


```python
class CIFAR_Image:
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()
        self.label = label

    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)
```

LMDB, karena menggunakan pemetaan memori, database baru perlu mengetahui seberapa besar memori yang akan digunakan. Ini relatif mudah dalam kasus ini, tetapi dapat menjadi sangat merepotkan dalam kasus lain, yang akan dilihat lebih dalam di bagian selanjutnya. LMDB menyebut variabel ini sebagai map_size. Operasi baca dan tulis dengan LMDB dilakukan dalam transaksi, yang dapat dianggap mirip dengan operasi pada database tradisional, terdiri dari sekelompok operasi pada database. Meskipun terlihat jauh lebih rumit daripada versi disk, tetapi teruskan dan terus membaca! Dengan tiga poin tersebut, mari kita lihat kode untuk menyimpan sebuah gambar tunggal ke LMDB.


```python
import lmdb
import pickle

def store_single_lmdb(image, image_id, label):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    map_size = image.nbytes * 10

    # Create a new LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), map_size=map_size)

    # Start a new write transaction
    with env.begin(write=True) as txn:
        # All key-value pairs need to be strings
        value = CIFAR_Image(image, label)
        key = f"{image_id:08}"
        txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()
```

LMDB selesai, sekarang lanjutkan untuk HDF5


```python
import h5py

def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()
```

**Experiment for Storing Single Image**

Sekarang dapat menggabungkan ketiga fungsi untuk menyimpan sebuah gambar tunggal ke dalam sebuah kamus, yang dapat dipanggil nanti selama eksperimen pengukuran waktu. Ini memungkinkan penggunaan fungsi-fungsi tersebut dengan lebih mudah dan terorganisir saat melakukan eksperimen untuk membandingkan kinerja berbagai metode.


```python
_store_single_funcs = dict(
    disk=store_single_disk, lmdb=store_single_lmdb, hdf5=store_single_hdf5
)
```

Akhirnya, semuanya siap untuk melakukan eksperimen pengukuran waktu. Mari coba menyimpan gambar pertama dari CIFAR beserta labelnya, dan menyimpannya dalam tiga cara yang berbeda. Ini akan memungkinkan untuk membandingkan kinerja ketiga metode penyimpanan yang telah dipersiapkan sebelumnya.


```python
from timeit import timeit

store_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_store_single_funcs[method](image, 0, label)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    store_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

    Method: disk, Time usage: 0.01844663499997523
    Method: lmdb, Time usage: 0.00705506700001024
    Method: hdf5, Time usage: 0.003116099999999733


1. Penyimpanan menggunakan metode HDF5 memiliki waktu penggunaan tercepat, dengan hanya membutuhkan waktu sekitar 0.003 detik.
2. Metode LMDB juga menunjukkan kinerja yang baik dengan waktu penggunaan sekitar 0.007 detik.
3. Meskipun masih efisien, penyimpanan menggunakan metode disk membutuhkan waktu yang lebih lama dibandingkan dengan LMDB dan HDF5, dengan waktu penggunaan sekitar 0.018 detik.



##**Storing Many Images**

Setelah melihat kode untuk menggunakan berbagai metode penyimpanan untuk menyimpan sebuah gambar, sekarang perlu menyesuaikan kode untuk menyimpan banyak gambar dan kemudian menjalankan eksperimen pengukuran waktu. Menyimpan beberapa gambar sebagai file .png adalah hal yang mudah dilakukan dengan memanggil store_single_method() beberapa kali. Namun, hal ini tidak berlaku untuk LMDB atau HDF5, karena Anda tidak ingin sebuah file basis data yang berbeda untuk setiap gambar. Sebaliknya, Anda ingin meletakkan semua gambar ke dalam satu atau lebih file. Perlu sedikit mengubah kode dan membuat tiga fungsi baru yang menerima banyak gambar, yaitu store_many_disk(), store_many_lmdb(), dan store_many_hdf5.


```python
def store_many_disk(images, labels):
    """ Stores an array of images to disk
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Save all the images one by one
    for i, image in enumerate(images):
        Image.fromarray(image).save(disk_dir / f"{i}.png")

    # Save all the labels to the csv file
    with open(disk_dir / f"{num_images}.csv", "w") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for label in labels:
            # This typically would be more than just one value per row
            writer.writerow([label])

def store_many_lmdb(images, labels):
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    map_size = num_images * images[0].nbytes * 10

    # Create a new LMDB DB for all the images
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), map_size=map_size)

    # Same as before — but let's write all the images in a single transaction
    with env.begin(write=True) as txn:
        for i in range(num_images):
            # All key-value pairs need to be Strings
            value = CIFAR_Image(images[i], labels[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()

def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()
```

Untuk dapat menyimpan lebih dari satu file ke disk, metode penyimpanan file gambar diubah untuk melakukan perulangan pada setiap gambar dalam daftar. Untuk LMDB, perulangan juga diperlukan karena kita membuat objek CIFAR_Image untuk setiap gambar dan metadatanya. Penyesuaian terkecil terjadi pada metode HDF5. Bahkan, hampir tidak ada penyesuaian sama sekali! File HDF5 tidak memiliki batasan ukuran file kecuali batasan eksternal atau ukuran dataset, sehingga semua gambar dimasukkan ke dalam satu dataset, sama seperti sebelumnya. Selanjutnya, perlu mempersiapkan dataset untuk eksperimen dengan meningkatkan ukurannya.

**Preparing The Dataset**

Sebelum menjalankan eksperimen lagi, pertama-tama perlu menggandakan ukuran dataset sehingga dapat melakukan pengujian hingga 100.000 gambar. Ini akan memungkinkan untuk memperluas cakupan eksperimen dan mendapatkan pemahaman yang lebih baik tentang kinerja metode penyimpanan yang berbeda.


```python
cutoffs = [10, 100, 1000, 10000, 100000]

# Let's double our images so that we have 100,000
images = np.concatenate((images, images), axis=0)
labels = np.concatenate((labels, labels), axis=0)

# Make sure you actually have 100,000 images and labels
print(np.shape(images))
print(np.shape(labels))
```

    (100000, 32, 32, 3)
    (100000,)


**Experiment for Storing Many Images**

Seperti yang dilakukan dengan membaca banyak gambar, dapat membuat kamus yang menangani semua fungsi dengan store_many_ dan menjalankan eksperimen. Ini memungkinkan untuk secara efisien menjalankan serangkaian eksperimen untuk membandingkan kinerja berbagai metode penyimpanan yang telah disiapkan sebelumnya. Dengan menggunakan kamus ini, dapat dengan mudah memanggil fungsi yang sesuai untuk setiap metode penyimpanan dan mengukur waktu yang diperlukan untuk menyimpan banyak gambar.


```python
_store_many_funcs = dict(
    disk=store_many_disk, lmdb=store_many_lmdb, hdf5=store_many_hdf5
)

from timeit import timeit

store_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_store_many_funcs[method](images_, labels_)",
            setup="images_=images[:cutoff]; labels_=labels[:cutoff]",
            number=1,
            globals=globals(),
        )
        store_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, Time usage: {t}")
```

    Method: disk, Time usage: 0.006477585999988378
    Method: lmdb, Time usage: 0.007818802000002734
    Method: hdf5, Time usage: 0.001811909999986483
    Method: disk, Time usage: 0.07950243799999157
    Method: lmdb, Time usage: 0.011568369999992001
    Method: hdf5, Time usage: 0.0029381999999884556
    Method: disk, Time usage: 0.6309681039999759
    Method: lmdb, Time usage: 0.060485386999999946
    Method: hdf5, Time usage: 0.005038743000000068
    Method: disk, Time usage: 4.788430167000001
    Method: lmdb, Time usage: 1.001933414000007
    Method: hdf5, Time usage: 0.09320840600000224
    Method: disk, Time usage: 57.18798056800003
    Method: lmdb, Time usage: 5.853249140000003
    Method: hdf5, Time usage: 0.9380266470000151


LMDB dan HDF5 memiliki kinerja yang lebih mangkus daripada PNG seperti plot yang ditunjukkan di bawah.


```python
import matplotlib.pyplot as plt

def plot_with_legend(
    x_range, y_data, legend_labels, x_label, y_label, title, log=False
):
    """ Displays a single plot with multiple datasets and matching legends.
        Parameters:
        --------------
        x_range         list of lists containing x data
        y_data          list of lists containing y values
        legend_labels   list of string legend labels
        x_label         x axis label
        y_label         y axis label
    """
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(10, 7))

    if len(y_data) != len(legend_labels):
        raise TypeError(
            "Error: number of data sets does not match number of labels."
        )

    all_plots = []
    for data, label in zip(y_data, legend_labels):
        if log:
            temp, = plt.loglog(x_range, data, label=label)
        else:
            temp, = plt.plot(x_range, data, label=label)
        all_plots.append(temp)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles=all_plots)
    plt.show()

# Getting the store timings data to display
disk_x = store_many_timings["disk"]
lmdb_x = store_many_timings["lmdb"]
hdf5_x = store_many_timings["hdf5"]

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Storage time",
    log=False,
)

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Log storage time",
    log=True,
)
```

    <ipython-input-16-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")



    
![png](120450092_Muhamad_Syahid_Burhanudien_Robbani_RC_files/120450092_Muhamad_Syahid_Burhanudien_Robbani_RC_37_1.png)
    


    <ipython-input-16-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")



    
![png](120450092_Muhamad_Syahid_Burhanudien_Robbani_RC_files/120450092_Muhamad_Syahid_Burhanudien_Robbani_RC_37_3.png)
    


##**Reading Single Image**

Selanjutnya akan dilakukan percobaan seberapa baik ketiga metode dalam membaca data yang sudah disimpan

Membaca dari disk


```python
def read_single_disk(image_id):
    """ Stores a single image to disk.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    image = np.array(Image.open(disk_dir / f"{image_id}.png"))

    with open(disk_dir / f"{image_id}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        label = int(next(reader)[0])

    return image, label
```

Membaca dengan lmdb


```python
def read_single_lmdb(image_id):
    """ Stores a single image to LMDB.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Encode the key the same way as we stored it
        data = txn.get(f"{image_id:08}".encode("ascii"))
        # Remember it's a CIFAR_Image object that is loaded
        cifar_image = pickle.loads(data)
        # Retrieve the relevant bits
        image = cifar_image.get_image()
        label = cifar_image.label
    env.close()

    return image, label
```

Membaca dengan hdf5


```python
def read_single_hdf5(image_id):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")

    image = np.array(file["/image"]).astype("uint8")
    label = int(np.array(file["/meta"]).astype("uint8"))

    return image, label
```

Membuat kamus untuk membaca semua fungsi


```python
_read_single_funcs = dict(
    disk=read_single_disk, lmdb=read_single_lmdb, hdf5=read_single_hdf5
)
```

Lakukan percobaan untuk Reading Single Image


```python
from timeit import timeit

read_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_read_single_funcs[method](0)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    read_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

    Method: disk, Time usage: 0.0018753209999999854
    Method: lmdb, Time usage: 0.0015632019999998192
    Method: hdf5, Time usage: 0.0020939799999837305


1. Penyimpanan menggunakan metode LMDB memiliki waktu penggunaan tercepat, dengan hanya membutuhkan waktu sekitar 0.0016 detik.
2. Metode disk juga menunjukkan kinerja yang baik dengan waktu penggunaan sekitar 0.0019 detik.
3. Meskipun masih efisien, penyimpanan menggunakan metode HDF5 membutuhkan waktu yang sedikit lebih lama dibandingkan dengan LMDB dan disk, dengan waktu penggunaan sekitar 0.0021 detik.

##**Reading Many Images**

Menambahkan dari fungsi di atas, dapat menambahkan fungsi read_many yang akan digunakan pada percobaan selanjutnya.


```python
def read_many_disk(num_images):
    """ Reads image from disk.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Loop over all IDs and read each image in one by one
    for image_id in range(num_images):
        images.append(np.array(Image.open(disk_dir / f"{image_id}.png")))

    with open(disk_dir / f"{num_images}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            labels.append(int(row[0]))
    return images, labels

def read_many_lmdb(num_images):
    """ Reads image from LMDB.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))
            # Remember that it's a CIFAR_Image object
            # that is stored as the value
            cifar_image = pickle.loads(data)
            # Retrieve the relevant bits
            images.append(cifar_image.get_image())
            labels.append(cifar_image.label)
    env.close()
    return images, labels

def read_many_hdf5(num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")

    return images, labels

_read_many_funcs = dict(
    disk=read_many_disk, lmdb=read_many_lmdb, hdf5=read_many_hdf5
)
```

Percobaan Many Images


```python
from timeit import timeit

read_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_funcs[method](num_images)",
            setup="num_images=cutoff",
            number=0,
            globals=globals(),
        )
        read_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")
```

    Method: disk, No. images: 10, Time usage: 7.310000000870787e-07
    Method: lmdb, No. images: 10, Time usage: 8.259999617621361e-07
    Method: hdf5, No. images: 10, Time usage: 5.630000146084058e-07
    Method: disk, No. images: 100, Time usage: 5.659999828822038e-07
    Method: lmdb, No. images: 100, Time usage: 4.930000159220072e-07
    Method: hdf5, No. images: 100, Time usage: 6.209999696693558e-07
    Method: disk, No. images: 1000, Time usage: 4.6400003839153214e-07
    Method: lmdb, No. images: 1000, Time usage: 3.9799999740353087e-07
    Method: hdf5, No. images: 1000, Time usage: 4.0399999079454574e-07
    Method: disk, No. images: 10000, Time usage: 4.309999894758221e-07
    Method: lmdb, No. images: 10000, Time usage: 5.090000172458531e-07
    Method: hdf5, No. images: 10000, Time usage: 4.940000053466065e-07
    Method: disk, No. images: 100000, Time usage: 3.89000035738718e-07
    Method: lmdb, No. images: 100000, Time usage: 3.9100001458791667e-07
    Method: hdf5, No. images: 100000, Time usage: 4.779999471793417e-07


1. Waktu penggunaan untuk menyimpan gambar cenderung berkurang seiring dengan peningkatan jumlah gambar untuk semua metode penyimpanan.
2. Metode LMDB dan HDF5 memiliki waktu penggunaan yang serupa, dengan metode LMDB sedikit lebih cepat untuk jumlah gambar yang lebih kecil, tetapi sebaliknya untuk jumlah gambar yang lebih besar.
3. Waktu penggunaan untuk metode disk juga cenderung berkurang seiring dengan peningkatan jumlah gambar, tetapi relatif stabil dan berada di tengah-tengah antara LMDB dan HDF5.

Seperti yang ditunjukkan pada grafik di bawah


```python
disk_x_r = read_many_timings["disk"]
lmdb_x_r = read_many_timings["lmdb"]
hdf5_x_r = read_many_timings["hdf5"]

plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to read",
    "Read time",
    log=False,
)

plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to read",
    "Log read time",
    log=True,
)
```

    <ipython-input-16-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")



    
![png](120450092_Muhamad_Syahid_Burhanudien_Robbani_RC_files/120450092_Muhamad_Syahid_Burhanudien_Robbani_RC_55_1.png)
    


    <ipython-input-16-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")



    
![png](120450092_Muhamad_Syahid_Burhanudien_Robbani_RC_files/120450092_Muhamad_Syahid_Burhanudien_Robbani_RC_55_3.png)
    


Perbandingan waktu menulis dan membaca dari ketiga metode.


```python
plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r, disk_x, lmdb_x, hdf5_x],
    [
        "Read PNG",
        "Read LMDB",
        "Read HDF5",
        "Write PNG",
        "Write LMDB",
        "Write HDF5",
    ],
    "Number of images",
    "Seconds",
    "Log Store and Read Times",
    log=False,
)
```

    <ipython-input-16-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")



    
![png](120450092_Muhamad_Syahid_Burhanudien_Robbani_RC_files/120450092_Muhamad_Syahid_Burhanudien_Robbani_RC_57_1.png)
    


Tampilan grafik memori yang digunakan dari ketiga metode

dapat dilihat bahwa LMDB dan HDF5 memakan lebih banyak memori dan hal ini patut menjadi pertimbangan apakah ukuran yang besar sebanding dengan kecepatan yang diberikan.


```python
# Memory used in KB
disk_mem = [24, 204, 2004, 20032, 200296]
lmdb_mem = [60, 420, 4000, 39000, 393000]
hdf5_mem = [36, 304, 2900, 29000, 293000]

X = [disk_mem, lmdb_mem, hdf5_mem]

ind = np.arange(3)
width = 0.35

plt.subplots(figsize=(8, 10))
plots = [plt.bar(ind, [row[0] for row in X], width)]
for i in range(1, len(cutoffs)):
    plots.append(
        plt.bar(
            ind, [row[i] for row in X], width, bottom=[row[i - 1] for row in X]
        )
    )

plt.ylabel("Memory in KB")
plt.title("Disk memory used by method")
plt.xticks(ind, ("PNG", "LMDB", "HDF5"))
plt.yticks(np.arange(0, 400000, 100000))

plt.legend(
    [plot[0] for plot in plots], ("10", "100", "1,000", "10,000", "100,000")
)
plt.show()
```


    
![png](120450092_Muhamad_Syahid_Burhanudien_Robbani_RC_files/120450092_Muhamad_Syahid_Burhanudien_Robbani_RC_59_0.png)
    


##**Kesimpulan**

Ketiga metode memiliki keunggulan dan kelebihannya masing-masing, dalam percobaan ini dapat dilihat bahwa LMDB dan HDF5 lebih mangkus dalam membaca dan menulis data, tetapi tidak dengan ukuran penyimpanannya. Maka dari itu perlu pemilihan metode yang bijak dalam menentukan mana yang lebih efektif sesuai kebutuhan.
