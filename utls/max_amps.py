import numpy as np
import random

# Генерируем тестовые данные: 5 эпох, 3 канала, 4 временных точки
data_stack = np.random.randint(0, 100, size=(5, 3, 4))
print("Исходные данные (форма):", data_stack.shape)
print(data_stack)

# max_amps: максимум по модулю для каждой эпохи (по времени и каналам)
max_amps = np.max(np.abs(data_stack), axis=(1, 2))
print("\nАмплитуды эпох:", max_amps)

# Сколько эпох убрать (10%)
n_drop = int(np.ceil(0.2 * data_stack.shape[0]))
print(f"\nНужно удалить {n_drop} эпох с максимальной амплитудой")

# Индексы эпох с наибольшими амплитудами
drop_idx = np.argsort(max_amps)[-n_drop:]
print("Индексы удаляемых эпох:", drop_idx)

# Маска эпох, которые оставляем (все, кроме самых «шумных»)
keep_mask = np.ones(data_stack.shape[0], dtype=bool)
keep_mask[drop_idx] = False

# Очищенные данные
data_clean = data_stack[keep_mask]
print(f"\nОчищенные данные (форма): {data_clean.shape}")
print(data_clean)