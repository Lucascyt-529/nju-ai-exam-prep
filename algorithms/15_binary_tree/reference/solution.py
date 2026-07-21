from collections import deque
class TreeNode:
    def __init__(self,value,left=None,right=None): self.value=value; self.left=left; self.right=right
def build_tree(tokens: list[str]):
    if not tokens or tokens[0]=="#": return None
    root=TreeNode(int(tokens[0])); queue=deque([root]); index=1
    while queue and index<len(tokens):
        node=queue.popleft()
        if tokens[index]!="#": node.left=TreeNode(int(tokens[index])); queue.append(node.left)
        index+=1
        if index<len(tokens) and tokens[index]!="#": node.right=TreeNode(int(tokens[index])); queue.append(node.right)
        index+=1
    return root
def level_order(root) -> list[list[int]]:
    if root is None: return []
    result=[]; queue=deque([root])
    while queue:
        row=[]
        for _ in range(len(queue)):
            node=queue.popleft(); row.append(node.value)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(row)
    return result
def main():
    tokens=input().split(); levels=level_order(build_tree(tokens)); print(" | ".join(" ".join(map(str,row)) for row in levels))
if __name__=="__main__": main()
