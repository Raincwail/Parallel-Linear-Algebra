# GEMV (Generic Matrix-Vector Product)

## 0. How to build
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

В итоге в папке build будет три бинарника: gemv_row, gemv_col, gemv_block

## 0. Запуск бенчмарков
```bash
./build/gemv_row.exe n_rows n_cols
```
где n_rows и n_cols - размеры генерируемой матрицы

## 1. Скрипт run.sh
Примитивный скрипт (WIP), который запускает замеры производительности (+ юнит тест) и записывает результаты в файл.
Применение:
```bash
run.sh ./path/to/executable n_rows n_cols
```

Например,
```bash
run.sh ./build/gemv_row.exe 5000 5000
```
`n_rows` и `n_cols` опциональны. По умолчанию используются значения 5000.

В результате в папке с билдом будут лежать два файла:
`executable_name_log_NxM.txt`,
`executable_name_times_NxM.txt`.
В первом содержится лог программы.
Во втором - время, затраченное на gemv через пробел по процессам.
Количество процессов задаётся перечислением в скрипте.


## 2. Замеры производительности
Время дано в миллисекундах

### Умножение по строкам:

| N процессов | 1000x1000 | 5000x5000 | 10000x10000 |
| ------------- | ------------- | ------------- | ------------- |
| 1 | 0.9 | 24.6 | 96 |
| 2 | 0.45 | 12 | 48 |
| 3 | 0.3 | 8 | 32 |
| 4 | 0.23 | 6.1 | 24.4 |
| 5 | 0.18 | 5.1 | 20.6 |
| 6 | 0.15 | 4.4 | 17.7 |
| 7 | 0.13 | 4.1 | 16.5 |
| 8 | 0.11 | 3.7 | 15.7 |

### Умножение по столбцам:

| N процессов | 1000x1000 | 5000x5000 | 10000x10000 |
| ------------- | ------------- | ------------- | ------------- |
| 1 | 0.63 | 164.1 | 782.5 |
| 2 | 0.32 | 82.6 | 397.5 |
| 3 | 0.21 | 57.9 | 268.2 |
| 4 | 0.16 | 44.5 | 211.6 |
| 5 | 0.15 | 38.5 | 178.9 |
| 6 | 0.11 | 32.6 | 164.1 |
| 7 | 0.09 | 26.9 | 165.1 |
| 8 | 0.08 | 25 | 161.5 |

### Умножение по блокам:

| N процессов | 1000x1000 | 5000x5000 | 10000x10000 |
| ------------- | ------------- | ------------- | ------------- |
| 1 | 0.88 | 24.4 | 97.6 |
| 4 | 0.23 | 6.5 | 25.7 |
| 9 | 0.11 | 4.3 | 16.8 |
| 16 | 0.06 | 16.8 | 15.2 |
