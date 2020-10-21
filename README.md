# Linear Regression with SGD implemented in Scala

- Принимает на вход csv файлы для трейна и теста. Данные в файлах boston-train.csv и boston-test.csv соответственно
- Результаты лежат в файлах out_train.csv и out_test.csv
- Процесс обучения по эпохам в файле linear_regression.log
- Обучается обычным SGD. Сверял с аналитическим решением для линейной регрессии, качество примерно такое же.

Запускать с параметрами:  
`--data-train boston-train.csv --data-test boston-test.csv --target-column 3 --out-train out_train.csv --out-test out_test.csv`
