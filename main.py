import multiprocessing
import os
import random
import albumentations as A
import cv2
import time
from multiprocessing import Process, Queue

# Lista con le trasformazioni disponibili
common_transformations = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomGamma(p=0.5),
    A.CLAHE(p=0.5),
    A.RandomCrop(width=200, height=200, p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
]

#Funzione per Thread
def generate_transformed_images(image, process_batch_number, images_queue=None):

        transformed_images = []

        for _ in range(process_batch_number):

            #Scelgo a caso un numero arbitrario tra 1 e 5 un numero di trasformazioni
            num_transforms = random.randint(1, 5)

            #Pesco a caso dalla lista di trasformazioni
            selected_transforms = random.sample(common_transformations, num_transforms)

            #Unisco tutte le trasformazioni
            transform = A.Compose(selected_transforms)

            #Applico le trasformazioni
            transformed_image = transform(image=image)["image"]


            if images_queue is not None:
                transformed_images.append(transformed_image)

        if images_queue is not None:
            images_queue.put(transformed_images)


if __name__ == "__main__":

        #NUMERO DI IMMAGINI DA GENERARE IN OUTPUT
        n = 10

        #True -> salva le immagini
        #False -> non salva le immagini
        saveImgs = True

        print("Numero di core Disponibili: ", multiprocessing.cpu_count())

        #carico l'immagine
        image = cv2.imread("input_image/paesaggio-grande.jpg")
        if image is None:
            raise ValueError("Immagine non trovata")

        #converto in RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        #Dal numero di CPU calcolo una divisione intera per distribuire il carico ai processi, il processo 0
        # farà il numero di uscita della divisione intera + resto della divisione.
        num_cpus = multiprocessing.cpu_count()
        process_batch_number = n // num_cpus
        process_batch_number_extra = n % num_cpus

        print("Ogni processo eseguirà :", process_batch_number, "immagini")
        print("Immagini extra da distribuire al processo 0:", process_batch_number_extra)

        #Alloca una coda se il salvataggio è abilitato
        if saveImgs:
            queue_images = Queue()
        else:
            queue_images = None


        processes = []

        start_time = time.time()

        # Faccio partire i processi
        for i in range(num_cpus):
            batch = process_batch_number + (process_batch_number_extra if i == 0 else 0)
            p = Process(target=generate_transformed_images,
                        args=(image.copy(), batch, queue_images))
            processes.append(p)
            p.start()

        # Raccolgo i risultati della coda
        all_transformed_images = []
        if saveImgs:
            for _ in range(num_cpus):
                try:
                    images = queue_images.get(timeout=30)  # 30 second timeout
                    all_transformed_images.extend(images)
                except Exception as e:
                    print(f"Errore nella coda: {e}")

        # Join sui processi
        for p in processes:
            p.join(timeout=1)

        print("Tempo di esecuzione:", time.time() - start_time, "secondi")

        # salva le immagini in "output_image"
        if saveImgs and all_transformed_images:
            os.makedirs("output_image", exist_ok=True)
            for i, img in enumerate(all_transformed_images):
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                output_path = os.path.join("output_image", f"transformed_{i}.jpg")
                cv2.imwrite(output_path, img_bgr)


