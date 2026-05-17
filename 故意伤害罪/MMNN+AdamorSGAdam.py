# -*- coding: utf-8 -*-
import os

from sympy.abc import alpha

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用同步错误报告
import torch.optim as optim
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt

lamda = 0
number = 10
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")


def save_results_excel(df, filename, subfolder=None):
    output_dir = OUTPUT_DIR if subfolder is None else os.path.join(OUTPUT_DIR, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")


noflee_lower = 6
noflee_upper = 36


flee_reduce_lower = 6
flee_reduce_upper = 36


flee_lower = 6
flee_upper = 36


flee_death_reduce_lower = 36
flee_death_reduce_upper = 120


flee_death_lower = 36
flee_death_upper = 120

fleeing_startpoint = flee_lower + (flee_upper - flee_lower) * lamda
death_fleeing_startpoint = flee_death_lower + (flee_death_upper - flee_death_lower) * lamda


device = torch.device('cpu')



class SaturatedClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lower, upper):
        ctx.save_for_backward(input, lower, upper)
        return torch.max(lower, torch.min(input, upper))  # 与原 output 一致

    @staticmethod
    def backward(ctx, grad_output):
        input, lower, upper = ctx.saved_tensors
        grad_input = grad_output.clone()

        # 梯度处理逻辑：
        inside = (input >= lower) & (input <= upper)       # 区间内梯度为1
        below = (input < lower)                            # 小于下限：削弱
        above = (input > upper)                            # 大于上限：削弱

        grad_input[inside] = grad_output[inside] * 1.0     # 正常反向传播
        grad_input[below] = grad_output[below] * 0.1       # 削弱梯度
        grad_input[above] = grad_output[above] * 0.1       # 削弱梯度

        return grad_input, None, None  #

class CustomModel(nn.Module):
    def __init__(self, base_feature_size, prior_feature_size, posterior_feature_size, NN_feature_size):
        super(CustomModel, self).__init__()
        # 随机初始化参数
        # self.theta1 = nn.Parameter(torch.tensor(20*np.random.rand(2), dtype=torch.float32, device=device))
        # self.theta2 = nn.Parameter(torch.tensor(-1+2*np.random.rand(prior_feature_size), dtype=torch.float32, device=device))
        # self.theta3 = nn.Parameter(torch.tensor(-1+2*np.random.rand(posterior_feature_size) , dtype=torch.float32, device=device))  # +1 常数项
        # delta=0.51log时L2SG初始参数:轻伤（用于预测）
        # self.theta1 = nn.Parameter(torch.tensor([0,0.007876181178010], dtype=torch.float32, device=device))
        # self.theta2 = nn.Parameter(torch.tensor([-0.026572879027145,-0.052552436123387,-0.128158605180144,-0.011569262721448,-0.041429809215716,-0.023718381571319,0,-0.004362744964366,0.022198759699418,-0.001783075138063,-0.133594175137010,0,0.055873446130931], dtype=torch.float32, device=device))
        # self.theta3 = nn.Parameter(torch.tensor([0,0,0.238235817468693,0.254697036181523,-0.0807555475483341,-0.0605712453768757,-0.0454048772493522,-0.0540328745249715,-0.287133068573654,0.104738365895258,0.0911270792299658,-0.0747479572036972,-0.0755860151586324,0.222695289139908,0.0370030052202306,0.0528338729700238,-0.0160913605541196,-0.0312609421774600,0.0819089698637200,-0.0142486444805108,0.00274146520537706,0.628125386505136], dtype=torch.float32, device=device))
        # # delta=0.51log时L2SG初始参数:重伤（用于预测）
        self.theta1 = nn.Parameter(torch.tensor([0.006053874192107,-0.014246674000886], dtype=torch.float32, device=device))
        self.theta2 = nn.Parameter(torch.tensor([-0.071460721748652,0,0.071999335873278,0.041459795735327,-0.089429572096502,0.035230012975787,0,0.023275538768743,0.010928107534989,0,-0.082097742167234,0,0], dtype=torch.float32, device=device))
        self.theta3 = nn.Parameter(torch.tensor([0.628234424950179,0.508776171673015,0.199507011046998,0.0351936995326325,-0.131493764273348,0.0391397386851738,-0.0499557203236371,-0.0856474377984399,-0.259742209776726,0.0415733433518542,-0.000231116326542458,-0.183433904285415,0.0933599811207825,0.0786486168767389,0.0505572262976934,-0.0170173130310525,0.00100343640699517,-0.0453841678053260,0.0729842596077152,-0.0677786868839388,-0.136214961497343,-1.55250548703934], dtype=torch.float32, device=device))


        # 添加三层全连接神经网络
        self.fc1 = nn.Linear(NN_feature_size, 128)  # 增加第一个隐藏层的神经元数量
        self.fc2 = nn.Linear(128, 128)  # 新增一个隐藏层
        self.fc3 = nn.Linear(128, 1)  # 输出大小为1


        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, a, injuries, prior, phi, NN):
        hiddenInput1 = a + torch.matmul(injuries, self.theta1)

        hiddenInput2 = 1 + prior * self.theta2
        hiddenInput2 = torch.prod(hiddenInput2, dim=1)

        hiddenInput3 = 1 + torch.matmul(phi, self.theta3)

        x = self.fc1(NN)
        x = self.relu(x)  # ReLU 激活
        x = self.fc2(x)
        x = self.relu(x)  # ReLU 激活
        x = self.fc3(x)   # 第三层全连接
        x = x.squeeze(1)

        inter_product = hiddenInput1 * hiddenInput2 * (hiddenInput3 + x)
        inter_product_unbias = hiddenInput1 * hiddenInput2 * hiddenInput3
        return inter_product, inter_product - x, x, inter_product_unbias



class RelativeAbsoluteErrorLoss(nn.Module):
    def __init__(self):
        super(RelativeAbsoluteErrorLoss, self).__init__()

    def forward(self, output, target):
        return torch.mean(torch.abs(output - target) / target)

def compute_regularization(model, mode='l1', lambda_=1e-4):
    penalty = 0
    for param in [model.fc1.weight, model.fc2.weight, model.fc3.weight]:
        if mode == 'l1':
            penalty += torch.sum(torch.abs(param))
        elif mode == 'l2':
            penalty += torch.sum(param ** 2)
    return lambda_ * penalty


def train_model(model, a_train, injuries_train, prior_train, phi_train, y_train, NN_train,
                is_1_train, is_2_train, is_3_train, is_4_train, is_5_train,
                optimizer, criterion, batch_size, max_epochs,
                a_test, injuries_test, prior_test, phi_test, y_test, NN_test,
                is_1_test, is_2_test, is_3_test, is_4_test, is_5_test,
                best_test_precision, best_model_state_dict,
                case_ids_test, province=None):

    train_accuracies = []
    total_seen_samples = 0  # ✅ 初始化累计样本数 k

    for epoch in range(max_epochs):
        model.train()
        train_size = a_train.shape[0]
        # permutation = torch.randperm(train_size, device=device) # 随机顺序
        permutation = torch.arange(train_size, device=device) # 固定顺序

        epoch_loss = 0.0
        precision_train = []

        for i in range(0, train_size, batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size].long()

            if (indices >= prior_train.size(0)).any():
                print(f"Invalid indices detected: {indices}")
                continue

            batch_a = a_train[indices]
            batch_injuries = injuries_train[indices]
            batch_prior = prior_train[indices]
            batch_phi = phi_train[indices]
            batch_y = y_train[indices]
            batch_NN = NN_train[indices]

            batch_is_1 = np.atleast_1d(is_1_train[indices])
            batch_is_2 = np.atleast_1d(is_2_train[indices])
            batch_is_3 = np.atleast_1d(is_3_train[indices])
            batch_is_4 = np.atleast_1d(is_4_train[indices])
            batch_is_5 = np.atleast_1d(is_5_train[indices])

            inter_product, mechanism, NN, inter_product_unbias = model(batch_a, batch_injuries, batch_prior, batch_phi, batch_NN)

            lower_bounds = torch.zeros_like(inter_product, device=device)
            upper_bounds = torch.zeros_like(inter_product, device=device)

            lower_bounds[batch_is_1 == 1] = noflee_lower
            upper_bounds[batch_is_1 == 1] = noflee_upper

            lower_bounds[batch_is_2 == 1] = flee_lower
            upper_bounds[batch_is_2 == 1] = flee_upper

            lower_bounds[batch_is_3 == 1] = flee_reduce_lower
            upper_bounds[batch_is_3 == 1] = flee_reduce_upper

            lower_bounds[batch_is_4 == 1] = flee_death_lower
            upper_bounds[batch_is_4 == 1] = flee_death_upper

            lower_bounds[batch_is_5 == 1] = flee_death_reduce_lower
            upper_bounds[batch_is_5 == 1] = flee_death_reduce_upper

            output = SaturatedClamp.apply(inter_product, lower_bounds, upper_bounds) # 反向传播考虑饱和函数
            # output = inter_product # 反向传播不考虑饱和函数

            # loss = criterion(output, batch_y)
            alpha_penalty = 1.4
            loss = criterion(output, batch_y)  + alpha_penalty*torch.abs(torch.mean(NN)+0.0370) # 限制偏置项大小：惩罚系数：轻伤0.4；重伤1.6;SG偏置项：轻伤-0.3392，重伤-0.0370 (NN:245)
            epoch_loss += loss.item()

            loss.backward()

            total_seen_samples += batch_a.size(0)  # ✅ 更新已处理样本数 k

            # ✅ 动态设置学习率为 0.001 / k
            if total_seen_samples > 0:
                current_lr =   0.001 # 1 / np.sqrt(total_seen_samples)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            optimizer.step()

            precision = 1 - torch.abs(batch_y - output) / batch_y
            precision_train.append(precision.detach().cpu().numpy())

        # ✅ 加上传递 case_ids_test 到 test_model
        current_test_precision, current_model_state_dict = test_model(
            model, a_test, injuries_test, prior_test, phi_test, y_test, NN_test,
            is_1_test, is_2_test, is_3_test, is_4_test, is_5_test,
            best_test_precision, best_model_state_dict,
            case_ids_test, province
        )

        if current_test_precision > best_test_precision:
            best_test_precision = current_test_precision
            best_model_state_dict = current_model_state_dict

        print(f'Epoch {epoch + 1}/{max_epochs}, Current Test Precision: {current_test_precision}, Best Test Precision: {best_test_precision}')

        precision_train = np.concatenate(precision_train)
        mean_precision_train = precision_train.mean()
        train_accuracies.append(mean_precision_train)
        print(f'Training Precision: {mean_precision_train}')

    return train_accuracies, best_test_precision, best_model_state_dict




def test_model(model, a_test, injuries_test, prior_test, phi_test, y_test, NN_test, is_1_test, is_2_test, is_3_test, is_4_test, is_5_test, best_test_precision, best_model_state_dict, case_ids_test, province=None):
    test_accuracies = []

    # 测试集评估
    model.eval()
    with torch.no_grad():
        inter_product_test, mechanism, NN, inter_product_unbias_test = model(a_test, injuries_test, prior_test, phi_test, NN_test)

        # 初始化 lower_bounds 和 upper_bounds
        lower_bounds = torch.zeros_like(inter_product_test, device=device)
        upper_bounds = torch.zeros_like(inter_product_test, device=device)

        # 设置饱和上下界
        lower_bounds[is_1_test == 1] = noflee_lower
        upper_bounds[is_1_test == 1] = noflee_upper

        lower_bounds[is_2_test == 1] = flee_lower
        upper_bounds[is_2_test == 1] = flee_upper

        lower_bounds[is_3_test == 1] = flee_reduce_lower
        upper_bounds[is_3_test == 1] = flee_reduce_upper

        lower_bounds[is_4_test == 1] = flee_death_lower
        upper_bounds[is_4_test == 1] = flee_death_upper

        lower_bounds[is_5_test == 1] = flee_death_reduce_lower
        upper_bounds[is_5_test == 1] = flee_death_reduce_upper


        output_test = torch.max(lower_bounds, torch.min(inter_product_test, upper_bounds))
        y_test_np = y_test.cpu().detach().numpy().flatten()
        inter_product_unbias_test_np = inter_product_unbias_test.cpu().detach().numpy().flatten()
        NN_np = NN.cpu().detach().numpy().flatten()
        # output_test = torch.round(output_test/6) / 2 * 12
        output_test_np = output_test.cpu().detach().numpy().flatten()

        # 精度计算
        precision_test = 1 - torch.abs(y_test - output_test) / y_test
        precision_test_np = precision_test.cpu().detach().numpy().flatten()
        mean_precision_test = precision_test.mean().item()
        test_accuracies.append(mean_precision_test)

        # 如果当前精度更高，则保存模型和案号 + 精度
        if mean_precision_test > best_test_precision:
            best_test_precision = mean_precision_test
            best_model_state_dict = model.state_dict()

            # 处理案号（转换为可用于 DataFrame 的格式）
            if torch.is_tensor(case_ids_test):
                case_ids_np = case_ids_test.cpu().detach().numpy()
            else:
                case_ids_np = np.array(case_ids_test)

            # 保存案号 + 精度到 Excel
            df = pd.DataFrame({
                'Case_ID': case_ids_np,
                'Prediction_Precision': precision_test_np,
                'output_test': output_test_np,
                'y_test':y_test_np,
                'inter_product_unbias_test': inter_product_unbias_test_np,
                'bias':NN_np
            })
            if province is not None:
                safe_province_name = province.replace(" ", "").replace("/", "_")
                save_results_excel(
                    df,
                    f"average_accuracy_MMNN_Adam_{safe_province_name}.xlsx",
                    "average_accuracy_MMNN_Adam_province"
                )
            else:
                save_results_excel(df, f"serious_Adam_{number}.xlsx")



    return best_test_precision, best_model_state_dict


def train_test_split(a, injuries, prior, phi, y, NN, is_1, is_2, is_3, is_4, is_5, case_ids):
    # 分割训练集和测试集
    train_size = round(a.shape[0] * 0.8)
    a_train, a_test = a[:train_size], a[train_size:]
    injuries_train, injuries_test = injuries[:train_size], injuries[train_size:]
    prior_train, prior_test = prior[:train_size], prior[train_size:]
    phi_train, phi_test = phi[:train_size], phi[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    NN_train, NN_test = NN[:train_size], NN[train_size:]
    is_1_train, is_1_test = is_1[:train_size], is_1[train_size:]
    is_2_train, is_2_test = is_2[:train_size], is_2[train_size:]
    is_3_train, is_3_test = is_3[:train_size], is_3[train_size:]
    is_4_train, is_4_test = is_4[:train_size], is_4[train_size:]
    is_5_train, is_5_test = is_5[:train_size], is_5[train_size:]
    case_ids_train, case_ids_test = case_ids[:train_size], case_ids[train_size:]

    return (a_train, a_test, injuries_train, injuries_test,
            prior_train, prior_test, phi_train, phi_test,
            y_train, y_test, NN_train, NN_test,
            is_1_train, is_1_test, is_2_train, is_2_test,
            is_3_train, is_3_test, is_4_train, is_4_test,
            is_5_train, is_5_test, case_ids_train, case_ids_test)

def feature_extract(table, plot_original, prior_features, base_features, posterior_features, NN_features):
    # #神经网络情节

    prison_normal = table['有期徒刑'].values

    # 提取索引
    prior_indices = [plot_original.index(attr) for attr in prior_features]
    base_indices = [plot_original.index(attr) for attr in base_features]
    posterior_indices = [plot_original.index(attr) for attr in posterior_features]
    # law_reduce_indices = [plot_original.index(feature) for feature in law_reduce_features]
    NN_indices = [plot_original.index(attr) for attr in NN_features]

    # 提取数据
    prior_data = table.iloc[:, prior_indices].values  # 先适用特征数据
    base_data = table.iloc[:, base_indices].values  # 基准刑情节数据
    posterior_data = table.iloc[:, posterior_indices].values  # 后适用特征数据
    NN_data = table.iloc[:, NN_indices].values  # 神经网络特征数据

    # 数据赋值和转换
    data_wu = table.values

    # 初始化回归向量和其他变量
    a = np.zeros(data_wu.shape[0])
    phi_t = np.zeros((posterior_data.shape[1], data_wu.shape[0]))
    injuries = np.zeros((data_wu.shape[0], 2))

    is_1 = np.zeros(data_wu.shape[0])
    is_2 = np.zeros(data_wu.shape[0])
    is_3 = np.zeros(data_wu.shape[0])
    is_4 = np.zeros(data_wu.shape[0])
    is_5 = np.zeros(data_wu.shape[0])

    for t in range(data_wu.shape[0]):
        x_k = base_data[t, :]


        is_5[t] = 0
        is_3[t] = 0
        is_2[t] = 0
        is_4[t] = 0
        is_1[t] = 0
        if x_k[0] > 0:
            a[t] = death_fleeing_startpoint
            is_4[t] = 1
        else:
            a[t] = fleeing_startpoint
            is_1[t] = 1

        if x_k[0] > 0:
            injuries[t, :] = np.array([(x_k[0] - 1), x_k[1]])
        else:
            injuries[t, :] = np.array([x_k[0], (x_k[1] - 1)])
        phi_t[:, t] = np.hstack( posterior_data[t, :])


    a = torch.tensor(a, dtype=torch.float32, device=device)
    injuries = torch.tensor(injuries, dtype=torch.float32, device=device)
    prior = torch.tensor(prior_data, dtype=torch.float32, device=device)
    phi = torch.tensor(phi_t.T, dtype=torch.float32, device=device)  # [num_samples, posterior_features +1]
    y = torch.tensor(prison_normal, dtype=torch.float32, device=device)
    NN = torch.tensor(NN_data, dtype=torch.float32, device=device)

    return a, injuries, prior, phi, y, NN, is_1, is_2, is_3, is_4, is_5


def fine_tune_provinces(provinces, table, test_table, overall_model_filename, parameters_folder, device, plot_original, prior_features, base_features, posterior_features, NN_features, max_epochs, batch_size, provinces_str):
    # 用于存储每个省份的最佳测试精度
    all_province_precisions = []
    province_names = []
    total_correct_predictions = 0
    total_test_samples = 0

    for province in provinces:
        print(f"Fine-tuning model for {province}...")

        # 全部省份数据和测试子集
        province_table = table[table['省份/直辖市'] == province]
        province_test_table = test_table[test_table['省份/直辖市'] == province]
        province_train_table = province_table.drop(index=province_test_table.index)

        # 提取训练集特征
        a_train, injuries_train, prior_train, phi_train, y_train, NN_train, \
        is_1_train, is_2_train, is_3_train, is_4_train, is_5_train = feature_extract(
            province_train_table, plot_original, prior_features, base_features, posterior_features, NN_features
        )

        # 提取测试集特征
        a_test, injuries_test, prior_test, phi_test, y_test, NN_test, \
        is_1_test, is_2_test, is_3_test, is_4_test, is_5_test = feature_extract(
            province_test_table, plot_original, prior_features, base_features, posterior_features, NN_features
        )

        # 提取测试集案号
        case_ids_test = province_test_table["案号"].tolist()

        model = CustomModel(
            base_feature_size=len(base_features),
            prior_feature_size=len(prior_features),
            posterior_feature_size=len(posterior_features),
            NN_feature_size=len(NN_features)
        ).to(device)
        model.load_state_dict(torch.load(overall_model_filename))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = RelativeAbsoluteErrorLoss()

        best_test_precision = 0
        best_model_state_dict = None

        train_accuracies, best_test_precision, best_model_state_dict = train_model(
            model, a_train, injuries_train, prior_train, phi_train, y_train, NN_train,
            is_1_train, is_2_train, is_3_train, is_4_train, is_5_train,
            optimizer, criterion, batch_size, max_epochs,
            a_test, injuries_test, prior_test, phi_test, y_test, NN_test,
            is_1_test, is_2_test, is_3_test, is_4_test, is_5_test,
            best_test_precision, best_model_state_dict, case_ids_test, province
        )

        print(f"Best test precision for {province}: {best_test_precision}")

        model_filename = os.path.join(parameters_folder, f"{province}_model.pth")
        torch.save(best_model_state_dict, model_filename)
        print(f"Model parameters for {province} saved to {model_filename}")

        all_province_precisions.append(best_test_precision)
        province_names.append(province)

        num_test_samples = len(y_test)
        num_correct_predictions = best_test_precision * num_test_samples
        total_correct_predictions += num_correct_predictions
        total_test_samples += num_test_samples
        print(total_test_samples)

    average_precision = total_correct_predictions / total_test_samples if total_test_samples > 0 else 0

    print("\nIndividual test precisions for all provinces:")
    for province, precision in zip(province_names, all_province_precisions):
        print(f"{province}: {precision}")

    print(f"\nAverage test precision for all provinces: {average_precision}")
    return None



def main():
    # , nrows =100
    table = pd.read_excel('merged_data_filtered_minor_0508_3.xlsx')
    # 主函数中统一划分后20%
    test_table = table.iloc[int(len(table) * 0.8):]

    case_ids = table["案号"].tolist()

    # 打乱数据集
    # table = table.sample(frac=1, random_state=42).reset_index(drop=True)
    # table = process(table)

    # 按省份划分数据集
    provinces = table['省份/直辖市'].unique()

    provinces_str = '_'.join(provinces)

    plot_original = table.columns.values.tolist()

    # 定义先适用特征、基准刑情节、后适用情节
    prior_features = ['16-18', '12-16', '75+' , '减轻刑事责任的精神病人' , '又聋又哑、盲人','防卫过当','紧急避险过当','犯罪预备','犯罪未遂','犯罪中止','从犯','胁从犯','教唆犯']
    # prior_features = [ '减轻刑事责任的精神病人','从犯']
    # 基准刑情节
    base_features = ['重伤人数', '轻伤人数']

    # law_reduce_features = ['积极赔偿', '和解', '谅解', '减轻']

    other_features = ['省份/直辖市', '案号', '判决时间', '有期徒刑','减轻']

    NN_original = table.columns.values.tolist()

    NN_features = [feature for feature in NN_original if feature not in other_features+base_features]


    posterior_features = [feature for feature in plot_original if
                          feature not in prior_features + base_features + other_features ]
    # 如果“缓刑”在 posterior_features 中，先移除它
    if '缓刑' in posterior_features:
        posterior_features.remove('缓刑')

    # 将“缓刑”添加到 posterior_features 的最后
    posterior_features.append('缓刑')

    print(posterior_features)

    a, injuries, prior, phi, y, NN, is_1, is_2, is_3, is_4, is_5 = feature_extract(table, plot_original, prior_features, base_features, posterior_features,  NN_features)

    a_train, a_test, injuries_train, injuries_test, prior_train, prior_test, phi_train, phi_test, y_train, y_test, NN_train, NN_test, is_1_train, is_1_test, is_2_train, is_2_test, is_3_train, is_3_test, is_4_train, is_4_test, is_5_train, is_5_test, case_ids_train, case_ids_test = train_test_split(a, injuries, prior, phi, y, NN, is_1, is_2, is_3, is_4, is_5, case_ids)

    # 训练总模型
    print("Training the overall model...")
    model = CustomModel(base_feature_size=len(base_features),
                        prior_feature_size=len(prior_features),
                        posterior_feature_size=len(posterior_features),
                        NN_feature_size=len(NN_features)).to(device)


    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.LBFGS(model.parameters(),
    #                               lr=0.1,  # 通常需要更大的学习率
    #                               max_iter=20,  # 每个批次的内部迭代次数
    #                               history_size=100)  # 历史信息存储量
    criterion = RelativeAbsoluteErrorLoss()

    # 超参数
    batch_size = 245
    max_epochs = 30
    max_epochs_province = 50


    best_test_precision = 0
    best_model_state_dict = None



    train_accuracies, best_test_precision, best_model_state_dict = train_model(model, a_train, injuries_train, prior_train, phi_train, y_train, NN_train,
                is_1_train, is_2_train, is_3_train, is_4_train, is_5_train, optimizer, criterion, batch_size, max_epochs, a_test, injuries_test, prior_test, phi_test, y_test, NN_test, is_1_test, is_2_test, is_3_test, is_4_test, is_5_test, best_test_precision, best_model_state_dict, case_ids_test)


    print(f"Best test precision for the overall model: {best_test_precision}")

    # 保存总模型参数
    overall_model_filename = f"serious_Adam_{number}.pth"
    torch.save(best_model_state_dict, overall_model_filename)
    print(f"Overall model parameters saved to {overall_model_filename}")

    # 创建总的省份模型参数文件夹
    parameters_folder = os.path.join(os.getcwd(), f"daihuanxing_parameters_biasinter_modify_serious_Adam")
    if not os.path.exists(parameters_folder):
        os.makedirs(parameters_folder)

    fine_tune_provinces(provinces, table, test_table, overall_model_filename, parameters_folder, device, plot_original,
                        prior_features, base_features, posterior_features, NN_features, max_epochs_province, batch_size, provinces_str)


    # ========== 计算所有样本的中间层输出并保存 ==========
    # print("Extracting hidden representations for ALL data...")
    #
    # # 恢复模型参数
    # model.load_state_dict(best_model_state_dict)
    # model.eval()  # 关闭 dropout 等训练特性
    #
    # # 构造 DataLoader 用于所有样本（训练+测试）
    # NN_all = np.vstack((NN_train, NN_test))
    # case_ids_all = case_ids_train + case_ids_test
    #
    # NN_tensor_all = torch.tensor(NN_all, dtype=torch.float32, device=device)
    # dataset_all = torch.utils.data.TensorDataset(NN_tensor_all)
    # dataloader_all = torch.utils.data.DataLoader(dataset_all, batch_size=batch_size, shuffle=False)
    #
    # intermediate_outputs_all = []
    #
    # with torch.no_grad():
    #     for batch in dataloader_all:
    #         NN_batch = batch[0]
    #
    #         x = model.fc1(NN_batch)
    #         x = model.relu(x)
    #         x = model.fc2(x)
    #         x = model.relu(x)
    #
    #         intermediate_outputs_all.append(x.cpu().numpy())
    #
    # # 拼接所有输出
    # all_hidden = np.vstack(intermediate_outputs_all)
    #
    # # 转为 DataFrame 并加入案号
    # df_hidden = pd.DataFrame(all_hidden)
    # df_hidden.insert(0, '案号', case_ids_all)
    #
    # # 保存到 Excel
    # df_hidden.to_excel('all_hidden_representation.xlsx', index=False)
    # print("All hidden representations saved to 'all_hidden_representation.xlsx'.")
    #
    #
    # # ========== 读取 \Gamma 和 b^{(3)} ==========
    # theta1 = model.theta1.data.cpu().numpy()
    # theta2 = model.theta2.data.cpu().numpy()
    # theta3 = model.theta3.data.cpu().numpy()
    # gamma = model.fc3.weight.data.cpu().numpy()  # shape: (1, 128)
    # b3 = model.fc3.bias.data.cpu().numpy()       # shape: (1,)
    # print("Gamma:", gamma)
    # print("b^{(3)}:", b3)
    # print("theta1:", theta1)
    # print("theta2:", theta2)
    # print("theta3:", theta3)

    return None


if __name__ == "__main__":
    main()
