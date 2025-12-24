import graphviz

def generate_mlp_architecture_diagram():
    # 根據你提供的 argparse 預設值
    hidden_sizes = [512, 256, 128]
    dropout_rate = 0.2

    # 初始化有向圖
    dot = graphviz.Digraph('MLP_Architecture', comment='The architecture of the MLP class provided')

    # 設定圖表整體樣式：從左到右排列 (LR)，使用正交線條
    dot.attr(rankdir='LR', splines='ortho', bgcolor='white')
    dot.attr('node', shape='box', style='filled', fillcolor='white', fontname='Arial')
    dot.attr('edge', fontname='Arial', color='#333333')

    # --- 定義節點 ---

    # 1. 輸入層
    dot.node('input', 'Input Features\n(Batch, in_dim)', shape='ellipse', fillcolor='#E3F2FD', color='#1565C0')

    last_output_node = 'input'

    # 2. 隱藏層迴圈 (對應程式碼中的 for width in hidden:)
    for i, width in enumerate(hidden_sizes):
        layer_num = i + 1
        cluster_name = f'cluster_hidden_{layer_num}'

        # 使用子圖 (Subgraph Cluster) 來分組 Linear -> ReLU -> Dropout
        with dot.subgraph(name=cluster_name) as c:
            c.attr(label=f'Hidden Block {layer_num}\n(Width: {width})', bgcolor='#F5F5F5', color='#9E9E9E', style='rounded')

            # 定義區塊內的節點名稱
            lin_node = f'h{layer_num}_linear'
            relu_node = f'h{layer_num}_relu'
            drop_node = f'h{layer_num}_dropout'

            prev_width = hidden_sizes[i-1] if i > 0 else "in_dim"
            
            c.node(lin_node, f'Linear Layer\n({prev_width} → {width})', fillcolor='#FFF9C4')
            c.node(relu_node, 'ReLU', shape='circle', width='0.6', fillcolor='#C8E6C9')
            c.node(drop_node, f'Dropout\n(p={dropout_rate})', style='filled,dashed', fillcolor='#FFECB3')

            # 連接區塊內的節點
            c.edge(lin_node, relu_node)
            c.edge(relu_node, drop_node)

        # 將上一個區塊的輸出連接到當前區塊的輸入(Linear)
        dot.edge(last_output_node, f'h{layer_num}_linear')
        # 更新最後一個輸出節點為當前區塊的 Dropout
        last_output_node = f'h{layer_num}_dropout'

    # 3. 輸出層 (最後一個 Linear)
    # 對應程式碼: layers.append(nn.Linear(prev, out_dim))
    final_lin_node = 'final_linear'
    dot.node(final_lin_node, f'Output Linear Layer\n({hidden_sizes[-1]} → out_dim)', fillcolor='#FFF9C4')
    dot.edge(last_output_node, final_lin_node)

    # 4. 最終輸出
    dot.node('output', 'Output Class Scores\n(Batch, out_dim)', shape='ellipse', fillcolor='#E3F2FD', color='#1565C0')
    dot.edge(final_lin_node, 'output')

    # 渲染並保存圖像 (需要安裝 graphviz 軟體)
    try:
        output_path = dot.render('mlp_architecture', format='png', cleanup=True)
        print(f"Architecture diagram saved to: {output_path}")
    except graphviz.backend.ExecutableNotFound:
        print("Graphviz executable not found. Please install Graphviz to generate the image file.")
        print("You can view the source code of the graph below:")
        print(dot.source)

if __name__ == '__main__':
    generate_mlp_architecture_diagram()