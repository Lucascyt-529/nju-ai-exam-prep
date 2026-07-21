"""学生练习：构造、反转并输出单链表。"""
class ListNode:
    def __init__(self, value: int, next_node=None): self.value=value; self.next=next_node
def build_linked_list(values: list[int]):
    raise NotImplementedError("请完成 build_linked_list")
def reverse_linked_list(head):
    raise NotImplementedError("请完成 reverse_linked_list")
def linked_list_values(head) -> list[int]:
    raise NotImplementedError("请完成 linked_list_values")
def main():
    n=int(input()); values=list(map(int,input().split())) if n else []; assert len(values)==n
    print(*linked_list_values(reverse_linked_list(build_linked_list(values))))
if __name__=="__main__": main()
