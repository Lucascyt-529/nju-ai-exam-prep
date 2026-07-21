import starter
def main():
    values=[1,2,3]; print("输入:",values,"期望: [3, 2, 1]")
    try: print("实际:",starter.linked_list_values(starter.reverse_linked_list(starter.build_linked_list(values))))
    except NotImplementedError as e: print("停止展示:",e)
if __name__=="__main__": main()
