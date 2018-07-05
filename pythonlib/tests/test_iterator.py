from doc_iter import *


def main():
    # Test:
    def f1(dict):
        return np.array(dict["k1"])

    def f2(dict):
        return np.array(dict["k2"])

    test_to_write = [{"sec": {"k1": [1, 10], "k2": [1, 2, 3]}},
                     {"sec": {"k1": [], "k2": [1, 2]}},
                     {"sec": {"k1": [3,19], "k2": [1]}},
                     {"sec": {"k1": [4], "k2": [1, 2, 3, 4]}},
                     {"sec": {"k1": [5,5], "k2": [1]}},
                     {"sec": {"k1": [6,6,6], "k2": [1, 2, 3]}},
                     {"sec": {"k1": [7], "k2": [1, 2, 3]}}]

    with open('test_to_write.json', 'w') as fout:
        for dic in test_to_write:
            json.dump(dic, fout)
            fout.write("\n")
    print("file written")

    # test_doc_iter = document_iterator('test_to_write.json', query_keys=["sec"], fns = [f1,f2])
    # for x in test_doc_iter:
    #     print(x)
    #
    # test_doc_iter = document_iterator('test_to_write.json', query_keys=["sec"], fns = [f1,f2])
    # print(list(grouper(3, test_doc_iter)))

    test_doc_iter = document_iterator('test_to_write.json', query_keys=["sec"], fns = [f1,f2])
    batch_iterator = BatchIterator(test_doc_iter, 3, [True, True], [1000, 100])
    batch_iter = batch_iterator.get()
    for batch in batch_iter:
        print (batch)

if __name__ == '__main__':
    main()

