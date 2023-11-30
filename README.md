# TicTacToe-RL
<br>
<h2>Задачи:</h2> 

- Разработать RL-агента;
- В качестве основы алгоритма RL использовать уравнение Беллмана;
- Агент ходит не однообразно;
- Добиться качественных статистических показателей игры агента:
  
  a) у агента невозможно выйграть;
  
  б) агент всегда ходит в пользу своей победы, сводя к максимально возможному минимуму процент ничьих.

<br>
<h2>Решение 1:</h2> 

<h4>Принцип RL - уравнение Беллмана:</h4>

<img width="420" alt="Снимок экрана 2023-11-30 в 03 57 14" src="https://github.com/WhiteSpite/TicTacToe-RL/assets/113059464/b908afe9-6a69-4d8a-a245-88db96edf9c2">

<h4>Основная проблема:</h4> 

Для какого следующего состояния игрового стола находить max Q (в целях обновления Q текущего хода)?:
1) Следующее состояние игрового стола, которое мы имеем после хода нашего агента брать некорректно, т.к. из него агент уже не сможет походить, т.к. ход переходит к противнику;
2) После хода нашего агента можно предсказывать ход противника на основе q таблицы нашего агента, и уже для состояния игрового стола, которое получится после теоритического хода противника, находить max Q;
3) Также можно попробовать находить Q max для состояния стола, к которому приводит наш ход, предварительно инвертировав это состояние (X заменить на O, а О заменить на X) стола и взяв Q max со знаком -. Логика в том, что чем больше значение Q max для состояния стола после нашего хода в глазах противника, тем менее ценным это состояние должно быть для нас.

<h4>Результаты:</h4> 

- Были испробованы все 3 метода, при каждом агент улучшал статистические показатели в процессе обучения, особенно при использовании 1-ого метода (вероятно потому что во 2-ом и 3-ем случае корректность определения Q max сильно зависит от хода противника, однако в 1-ом случае Q распространяется некорректно в результате логической ошибки упомянутой выше);
- Однако результаты были не идеальны и сильно зависели от уровня рандомности агента-противника;
- Игра с изменением конфигурации (gamma, alpha, epsilon, награды за победу, поражение, ничью и ход) не привела к желаемым результатам;
- Принято решение попробовать иные алгоритмы RL обучения.

<br>
<h2>Решение 2:</h2> 

<h4>Принцип RL:</h4> 

- Агент запоминает все сделанные им ходы;
- После окончания игры агент обновляет ценность этих ходов соответственно наградам за победу, поражение или ничью.

<h4>Результаты:</h4>

- При alpha < 1, а также при динамически снижаемом alpha, агент не показал идеальные статистические показатели в игре с рандомом;
- При alpha == 1 агент научился показывать идеальные статистические результаты с противником любого уровня рандомности;
- Единственный недостаток - агент ходит в большинстве случаев однообразно, и существенно повлиять на это не получилось;
- Принято решение попробовать иные алгоритмы RL обучения.

<h4>График обучения QAgent (alpha == 1) при игре с противником который ходит рандомно:</h4>

![RL_vs_random](https://github.com/WhiteSpite/TicTacToe-RL/assets/113059464/bcbc8b11-a7b7-4d4e-91b5-c15d31e07fbe)

Из графика можно видеть, что для полного обучения агенту достаточно провести ~8823 игры.
Достигнутая статистика:
- Победы: 94%;
- Поражения: 0%;
- Ничьи: 5%;

<br>
<h2>Решение 3:</h2>

<h4>Принцип RL:</h4> 

- Пока обучение не завершено, наш агент ходит исключительно с epsilon равным 1;
- Пока обучение не завершено, наш агент играет исключительно с развитым противником с epsilon равным ~0.7;
- К каждому из возможных ходов по всей Q таблице агента пристваиваются три счетчика для побед, поражений и ничьих соответственно;
- Агент запоминает все сделанные им ходы;
- После окончания игры агент увеличивает соответствующий результату игры счетчик на 1 для каждого из запомненных ходов;
- Агент учится на максимально возможном количестве игр;
- После окончания обучения в качестве Q каждого из ходов в Q таблице устанавливается среднее арифмитическое между счетчиками соответствующего хода;
- Затем выполняется кластеризация (метод k-средних) Q всех ходов в Q таблице на минимально возможное (не влияющее на статистику агента) количество кластеров (в целях минимизации однообразности игры);
- В качестве новых Q для каждого из ходов в Q таблице устанавливаются лейблы соответствующих кластеров.

<h4>Результаты:</h4> 

- Агент научился показывать идеальные статистические результаты с противником любого уровня рандомности;
- Агент ходит не однообразно;

<h4>График игры обученного MeansQAgent при игре с противником который ходит рандомно:</h4>

![ScreenFlow2](https://github.com/WhiteSpite/TicTacToe-RL/assets/113059464/9e6c4cff-5914-40c7-af19-490d05580dcd)

Агент был обучен на 1млн игр. Q всех ходов Q таблицы были кластеризованы на 22 кластера. Достигнутая статистика:
- Победы: 96%;
- Поражения: 0%;
- Ничьи: 3%;

<br>
<h2>Выводы:</h2>

- Были испробованы различные варианты RL обучения в целях создания совершенного агента. Самым эффективным решением оказалось 3-тье: накопление информации в счетчиках, нахождение среднего арифметического и кластеризация Q;
- Использование уравнения Беллмана оказалось не лучшим подходом для обучения агента игре в крестики-нолики;

<br>
<h2>Дополнительно:</h2>

- Агент самостоятельно создает Q таблицу и дополняет ее новыми состояниями стола по ходу игры, также инициализируя все возможные ходы для этих состояний; 
- Был реализован алгоритм отождествления эквивалентных состояний стола (перевороты стола, отзеркаливание и символьная инверсия). Это позволило сократить количество обрабатываемых состояний стола в 16 раз (в итоге агент знает всего 628 возможных состояний);
- Был реализован удаленный агент на сервере. Пример работы с его API показан в api_request_example.py.
