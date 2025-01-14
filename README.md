# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/gaspardpetit/verbatim/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                            |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| verbatim/\_\_init\_\_.py                        |       10 |        0 |        0 |        0 |    100% |           |
| verbatim/audio/\_\_init\_\_.py                  |        0 |        0 |        0 |        0 |    100% |           |
| verbatim/audio/audio.py                         |       86 |       61 |       32 |        3 |     24% |21-28, 33, 43-47, 53-77, 101-128, 133-135 |
| verbatim/audio/sources/\_\_init\_\_.py          |        0 |        0 |        0 |        0 |    100% |           |
| verbatim/audio/sources/audiosource.py           |       31 |        4 |        0 |        0 |     87% |25, 29, 33, 44 |
| verbatim/audio/sources/factory.py               |       74 |       30 |       30 |        7 |     55% |65-67, 76-78, 84, 117->145, 119, 129-137, 145->150, 171-216 |
| verbatim/audio/sources/ffmpegfileaudiosource.py |       90 |       14 |       26 |        9 |     78% |41, 56-61, 67, 80-81, 85, 111-115, 132, 142, 150->153 |
| verbatim/audio/sources/fileaudiosource.py       |       84 |       27 |       18 |        6 |     62% |29, 32-37, 49, 56, 60, 70, 97-99, 105-121 |
| verbatim/audio/sources/micaudiosource.py        |       90 |       90 |       14 |        0 |      0% |     1-147 |
| verbatim/audio/sources/pcmaudiosource.py        |       42 |       42 |        4 |        0 |      0% |      1-71 |
| verbatim/audio/sources/sourceconfig.py          |       10 |        0 |        0 |        0 |    100% |           |
| verbatim/audio/sources/wavsink.py               |       18 |        0 |        2 |        0 |    100% |           |
| verbatim/config.py                              |       87 |       13 |       30 |       13 |     74% |189-198, 203, 205->208, 209, 211, 214, 222->224, 224->226, 226->228, 228->230, 230->232, 237, 243, 246 |
| verbatim/eval/\_\_init\_\_.py                   |        0 |        0 |        0 |        0 |    100% |           |
| verbatim/eval/cli.py                            |       17 |       17 |        0 |        0 |      0% |      1-27 |
| verbatim/eval/metrics.py                        |      173 |       26 |       58 |       13 |     81% |21-44, 49-58, 63-69, 131, 155, 158, 166, 168, 197, 201, 213, 236->239, 238, 251, 279->273, 292->296 |
| verbatim/eval/utils.py                          |      239 |      179 |       96 |        1 |     19% |56, 64-70, 81-109, 114-142, 148-158, 166, 180, 195-204, 209-233, 239, 256-263, 267-268, 272-273, 277-301, 308-353, 365-382, 387-390, 409-422, 427-429, 444-459, 473-486 |
| verbatim/main.py                                |      158 |      158 |       54 |        0 |      0% |     1-379 |
| verbatim/models.py                              |       32 |        9 |        6 |        2 |     66% |24-34, 46-48 |
| verbatim/transcript/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100% |           |
| verbatim/transcript/format/\_\_init\_\_.py      |        0 |        0 |        0 |        0 |    100% |           |
| verbatim/transcript/format/ass.py               |      206 |      206 |       80 |        0 |      0% |     5-528 |
| verbatim/transcript/format/docx.py              |      113 |      113 |       46 |        0 |      0% |     1-215 |
| verbatim/transcript/format/json.py              |       34 |       34 |        4 |        0 |      0% |      1-74 |
| verbatim/transcript/format/json\_dlm.py         |       37 |        0 |        6 |        1 |     98% |  35->exit |
| verbatim/transcript/format/md.py                |      148 |      148 |       76 |        0 |      0% |     1-263 |
| verbatim/transcript/format/multi.py             |       22 |       22 |        8 |        0 |      0% |      1-39 |
| verbatim/transcript/format/stdout.py            |        8 |        8 |        0 |        0 |      0% |      1-24 |
| verbatim/transcript/format/txt.py               |       71 |       23 |       14 |        1 |     62% |63-65, 91-96, 99, 102, 105, 113-124, 129, 138, 141 |
| verbatim/transcript/format/writer.py            |       47 |        3 |        0 |        0 |     94% |56, 60, 69 |
| verbatim/transcript/formatting.py               |        6 |        0 |        0 |        0 |    100% |           |
| verbatim/transcript/idprovider.py               |       18 |        2 |        0 |        0 |     89% |     6, 10 |
| verbatim/transcript/sentences.py                |       38 |       20 |       10 |        0 |     38% | 14, 20-44 |
| verbatim/transcript/transcript.py               |        5 |        5 |        0 |        0 |      0% |       1-7 |
| verbatim/transcript/words.py                    |       40 |        5 |        0 |        0 |     88% |34-36, 55, 58 |
| verbatim/verbatim.py                            |      459 |       81 |      174 |       36 |     79% |44, 62->68, 64, 69->75, 118, 120, 160, 174, 184-188, 200->202, 239-247, 258, 288, 291->269, 301, 320-321, 340, 373-375, 411, 427, 446, 473, 475->481, 484-491, 494-497, 501, 508, 540->539, 545-553, 579, 638->562, 661-662, 674-676, 685->693, 688-691, 694-696, 706-707, 710-712, 719, 732-738, 749-753 |
| verbatim/voices/\_\_init\_\_.py                 |        0 |        0 |        0 |        0 |    100% |           |
| verbatim/voices/diarization.py                  |       30 |       10 |        4 |        2 |     65% |18-20, 23, 26-27, 32, 37, 55-57 |
| verbatim/voices/diarize/base.py                 |       11 |        0 |        0 |        0 |    100% |           |
| verbatim/voices/diarize/factory.py              |       12 |        4 |        6 |        2 |     56% | 23, 25-28 |
| verbatim/voices/diarize/pyannote.py             |       22 |        0 |        4 |        2 |     92% |16->exit, 36->39 |
| verbatim/voices/diarize/stereo.py               |       56 |       43 |       18 |        0 |     18% |16, 19-22, 25-29, 38-85 |
| verbatim/voices/isolation.py                    |       44 |       29 |       10 |        0 |     28% |18-19, 22, 25-26, 29-44, 47-79 |
| verbatim/voices/separation.py                   |       58 |       58 |       16 |        0 |      0% |     1-122 |
| verbatim/voices/silences.py                     |       21 |        1 |        0 |        0 |     95% |        23 |
| verbatim/voices/transcribe/\_\_init\_\_.py      |        0 |        0 |        0 |        0 |    100% |           |
| verbatim/voices/transcribe/faster\_whisper.py   |       51 |       12 |       22 |        4 |     67% |25->28, 36, 53-62, 80 |
| verbatim/voices/transcribe/transcribe.py        |       26 |        2 |        0 |        0 |     92% |    30, 47 |
| verbatim/voices/transcribe/whisper.py           |       61 |       61 |       14 |        0 |      0% |     1-143 |
| verbatim/voices/transcribe/whispercpp.py        |       40 |       40 |        8 |        0 |      0% |     3-101 |
| verbatim/voices/transcribe/whispermlx.py        |       59 |       59 |       20 |        0 |      0% |     3-148 |
|                                       **TOTAL** | **2984** | **1659** |  **910** |  **102** | **40%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/gaspardpetit/verbatim/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/gaspardpetit/verbatim/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/gaspardpetit/verbatim/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/gaspardpetit/verbatim/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fgaspardpetit%2Fverbatim%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/gaspardpetit/verbatim/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.