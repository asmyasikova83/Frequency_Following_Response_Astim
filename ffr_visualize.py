import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Путь к директории с картинками
output_dir = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\pics\step2\pics_to_combine'

# Путь для сохранения объединённой картинки
output_path = os.path.join(output_dir, 'pics_combined.png')

# Получаем список всех файлов с изображениями в директории
image_extensions = ('.png')
image_files = [
    f for f in os.listdir(output_dir)
    if f.lower().endswith(image_extensions)
]

if not image_files:
    print("В указанной директории не найдено изображений.")
else:
    # Разделяем файлы на две группы: с суффиксом 'filt' и без
    filt_images = [f for f in image_files if 'filt' in f.lower()]
    other_images = [f for f in image_files if 'filt' not in f.lower()]

    # Сортируем обе группы для предсказуемого порядка
    filt_images.sort()
    other_images.sort()

    # Объединяем: сначала картинки с 'filt', затем остальные
    sorted_image_files = filt_images + other_images

    print(f"Найдено изображений: {len(sorted_image_files)}")
    print("Файлы в порядке отображения:", sorted_image_files)

    # Загружаем изображения в новом порядке
    images = []
    for img_file in sorted_image_files:
        img_path = os.path.join(output_dir, img_file)
        img = mpimg.imread(img_path)
        images.append(img)
        print(f"Загружено: {img_file}")
    n_images = len(images)

    # Создаём фигуру с одним столбцом (все картинки друг под другом)
    fig, axes = plt.subplots(n_images, 1, figsize=(8, n_images * 4))  # уменьшили множитель для высоты

    for idx, ax in enumerate(axes):
        ax.imshow(images[idx])
        # Подсвечиваем светло‑зелёным, если файл содержит 'filt'
        ax.axis('off')  # Убираем оси

    #plt.suptitle('Объединённые изображения (в один столбец)', fontsize=16)
    plt.tight_layout()

    # Сохраняем объединённую фигуру
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nОбъединённая картинка сохранена: {output_path}")
    plt.show()
