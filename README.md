# Parallel-Linear-Algebra
[Более подробное описание задачи](https://github.com/Raincwail/Parallel-Linear-Algebra/blob/master/task.pdf)

## 3.  С использованием CUDA решить СЛАУ прямым (Гаусс) или итерационным методом (CG, сопряженные градиенты).

### Описание задачи

На вход подается симметричная, положительно определенная матрица $A$ и вектор $b$, в качестве результат алгоритм выдает решение СЛАУ $Ax = b$.

Алгоритм представляет из себя итеративный градиентный спуск с минимизацией  $\phi(x) = \frac{1}{2}x^T A x - x^T b$.

### Решение задачи

В качестве решения были реализованы [итеративная](https://github.com/Raincwail/Parallel-Linear-Algebra/tree/master/CG-CUDA/iterative) и [параллельная](https://github.com/Raincwail/Parallel-Linear-Algebra/tree/master/CG-CUDA/parallel) версии.

Для установки матрицы $A$ и вектора $b$ используется заголовок [Base](https://github.com/Raincwail/Parallel-Linear-Algebra/blob/master/CG-CUDA/Base.h).

Основное отличие заключается в использованных векторных операциях - итеративная версия использует наивные имплементации, в то время как параллельная использует инструменты CUDA для ускорения вычислений.

### Оценка качества

В качестве оценки качества решения были проведены следующие эксперименты и получены соответствующие результаты (мс).

| Размер Матрицы / Версия Решения | 512x512 | 1024x1024 | 2048x2048 | 4096x4096 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Итеративная версия | 639 | 2518 | 9892 | 38988 |
| Параллельная версия (16 потоков) | 89 | 194 | 548 | 1470 |
| Параллельная версия (32 потока) | 77 | 149 | 325 | 752 |
| Параллельная версия (64 потока) | 92 | 127 | 307 | 745 |

Для замеров времени использовались [Timer](https://github.com/Raincwail/Parallel-Linear-Algebra/blob/master/CG-CUDA/iterative/Timer.h) и [GpuTimer](https://github.com/Raincwail/Parallel-Linear-Algebra/blob/master/CG-CUDA/parallel/GpuTimer.cuh) соответственно.

Более наглядные графики, а также таблицы анализа ускорения и эффективности можно найти в ноутбуке [analysis](https://github.com/Raincwail/Parallel-Linear-Algebra/blob/master/CG-CUDA/analysis.ipynb).
