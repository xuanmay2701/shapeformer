## ShapeFormer: Shapelet Transformer for Multivariate Time Series Classification



### Shapelet discovery
```
python cpu_main.py --dataset_pos=[dataset_pos] --num_shapelet=[num_shapelet] --window_size=[window_size]
```

### Model training
```
python main.py --dataset_pos=[dataset_pos] --num_shapelet=[num_shapelet] --window_size=[window_size]
```

Here, [dataset_pos], [num_shapelet] and [window_size] can be selected as follows:

| Dataset                   | [dataset_pos] | [window_size] | [num_shapelet] |
|---------------------------|---------------|---------------|----------------|
| ArticularyWordRecognition | 0             | 50            | 10             |
| AtrialFibrillation        | 1             | 100           | 3              |
| BasicMotions              | 2             | 100           | 10             |
| CharacterTrajectories     | 3             | 50            | 3              |
| Cricket                   | 4             | 200           | 30             |
| DuckDuckGeese             | 5             | 10            | 100            |
| ERing                     | 6             | 50            | 100            |
| EigenWorms                | 7             | 10            | 10             |
| Epilepsy                  | 8             | 20            | 30             |
| EthanolConcentration      | 9             | 200           | 100            |
| FaceDetection             | 10            | 10            | 10             |
| FingerMovements           | 11            | 20            | 30             |
| HandMovementDirection     | 12            | 200           | 100            |
| Handwriting               | 13            | 20            | 30             |
| Heartbeat                 | 14            | 200           | 100            |
| InsectWingbeat            | 15            | 10            | 30             |
| JapaneseVowels            | 16            | 10            | 1              |
| LSST                      | 17            | 20            | 10             |
| Libras                    | 18            | 10            | 30             |
| MotorImagery              | 19            | 100           | 30             |
| NATOPS                    | 20            | 20            | 1              |
| PEMS-SF                   | 21            | 50            | 10             |
| PenDigits                 | 22            | 4             | 10             |
| PhonemeSpectra            | 23            | 20            | 30             |
| RacketSports              | 24            | 10            | 10             |
| SelfRegulationSCP1        | 25            | 100           | 100            |
| SelfRegulationSCP2        | 26            | 100           | 100            |
| SpokenArabicDigits        | 27            | 10            | 100            |
| StandWalkJump             | 28            | 10            | 100            |
| UWaveGestureLibrary       | 29            | 10            | 10             |


