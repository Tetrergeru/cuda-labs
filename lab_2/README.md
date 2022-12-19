# Результаты профилировки Lab1

## Копирование данных

Скорость копирования с хоста на девайс: **10.282 GB/s**

У компьютера, которым я пользовался для тестов видоарта подключена к слоту с `PCI 3.0`: 
- Transfer rate per lane: **8.0 GT/s**
- Width: **x16**
- Throughput: **15.754 GB/s**

Получается, что реальный thrpughput ниже реального. Видимо видеокарта, процессор или память не справляется.

## Работа kernel-функции

Объединения блоков не происходит.

Device Memory Read Throughput: **8,47 GB/s**
Device Memory Write Throughput: **10,48 GB/s**

Device Memory utilization: **High (9/10)**

Время работы: **751 micro sec**
Использование регистров: **11 per thread**
Теоретическая загруженность: **100%**
Фактическая загруженность: **86.4%**

Issued IPC: **2.0617**
Executed IPC: **3.0636**

Global Load Efficiency: **33.3%**
Global Store Efficiency: **24.7%**

L1 Hit Rate: **62%**
L2 Read Hit Rate: **66%**
L2 Write Hit Rate: **85%** 

## Возможности оптимизации

Возможности для оптимизации в два и более раза не вижу
