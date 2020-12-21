# Sudoku

探索: Lecun
超参数
正则化


## Demo
可以用如下代码运行 demo
```
python solve_sudoku_puzzle.py -m output/digit_classifier.h5 -i sudoku_puzzle.jpg
```
Notice that the programme is used on a server so `cv2.show` is not allowed.

## Structure

```
SUDOKU/
|--opencv-sudoku-solver/:
|   |--output/: Digit model storage
|   |--pyimagesearch/:
|       |--models/:
|       |--sudoku/:
|       |--__init__.py/:
|   |--solve_sudoku_puzzle.py/:
|   |--train_digit_classifier/:
|
|--requirement.txt: Dependent library list
|--README.md
```

## Plan
Sudoku_solver 的思路:
1. 首先 (img, warped) = find_puzzle, 找到 puzzle 在原来图像中的位置, 返回彩色和黑白的修正图像
2. 定义棋盘 `board = np.zeros((9, 9), dtype="int")`
3. 把 warped 划分为 9x9 的 cells, 用 model.predict 得到 cell 中对应的数字并填写到 board 中
4. 调用 Class puzzle `puzzle = Sudoku(3, 3, board=board.tolist())`, 但是追溯下去我发现真正 solve 的过程发生在
函数 `_SudokuSolver -> __get_solution`
5. 最后输出什么的都是细枝末节

### _SudokuSolver -> __get_solution
研究该函数采用了什么方法解决了 sudoku, 从而做到对 correction 的实现.
发现竟然在用穷举的方法做, 非常低效、愚蠢, 但是不得不说, 这就是一个标准的 CSP 解法.

我们如何在此基础上做 error correction 呢?
如果对每个 given cell 的值一次进行遍历试错的话, 代价就太高了. 如果非要采用这个思路, 我们需要更快的速度解决 sudoku 的方法.

另一种思路是对识别出来的把握最低的数字进行修正, 比如把该数字换成概率第二大的数字. 然后对新 puzzle 进行求解.

### Train SudokuNet
SudokuNet 是一个图像分类网络:
1. 输入大小为 28x28 的手写数字
2. Classes: {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
            '一':10, '二':11, '三':12, '四':13, '五':14, '六':15, '七':16, '八':17, '九':18, '十':19}

关于这个训练我有一些要说, 就是 minist 和中文手写数据集不能放在一起训练.
由于数据量不平衡, 一定会出现准确率不平衡的情况.

所以虽然是一个模型, 但是仍然应该分开训练.


我们可不可能一次只训练数据中的某一类呢?
