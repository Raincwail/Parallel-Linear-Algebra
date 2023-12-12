# GEMV (Generic Matrix-Vector Product)

## 0. How to build
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

В итоге в папке build будет три бинарника: gemv_row, gemv_col, gemv_block

## 1. Скрипт run.sh
Примитивный скрипт (WIP), который запускает замеры производительности (+ юнит тест) и записывает результаты в файл
Применение:
```bash
run.sh ./path/to/executable n_rows n_cols
```

Например,
```bash
run.sh ./build/gemv_row.exe 5000 5000
```

Результирующий файл с замерами по 9 процессам будет лежать рядом с executable файлом

## 2. Реализовать алгоритмы для умножения матрицы на вектор, используя разбиение по строкам, по столбцам и по блокам.
TODO

### Описание задачи

TODO

### Решение задачи

TODO

### Оценка качества

TODO
