"""学生练习：按层序编码构造二叉树并输出层序遍历。"""
class TreeNode:
    def __init__(self,value,left=None,right=None): self.value=value; self.left=left; self.right=right
def build_tree(tokens: list[str]):
    raise NotImplementedError("请完成 build_tree")
def level_order(root) -> list[list[int]]:
    raise NotImplementedError("请完成 level_order")
def main():
    tokens=input().split(); levels=level_order(build_tree(tokens)); print(" | ".join(" ".join(map(str,row)) for row in levels))
if __name__=="__main__": main()
