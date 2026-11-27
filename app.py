# -*- coding: utf-8 -*-
"""获卵数多粒度智能预测系统 — 交互式网页计算器
模型: 5任务堆叠(共享9变量组12列), 训练队列 n=9,942 (step200, 与论文step186同口径)
运行: streamlit run app.py
"""
import os
import json
import threading
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='获卵数多粒度智能预测系统',
                   page_icon='🧬', layout='wide', initial_sidebar_state='expanded')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_oocyte_stack.joblib')

BIN_A_LBL = ['<1枚', '1-3枚', '4-7枚', '8-12枚', '13-20枚', '21-29枚', '≥30枚']
BIN_B_LBL = ['<1枚', '1-5枚', '6-10枚', '11-18枚', '19-25枚', '≥26枚']

# ---------------- 模型加载与预测 ----------------
META_PATH = os.path.join(BASE_DIR, 'meta.json')

@st.cache_data(show_spinner=False)
def load_meta():
    with open(META_PATH, encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

_LOAD_LOCK = threading.Lock()

def get_model():
    """串行化加载: 后台预热线程与预测请求不会重复读盘"""
    with _LOAD_LOCK:
        return load_model()

def stack_predict(task, X):
    """X: (n, 12) ndarray, 顺序=MODEL['features']"""
    names, fold_models, kind = task['names'], task['fold_models'], task['kind']
    Z = np.zeros((X.shape[0], len(names)))
    for j, fmods in enumerate(fold_models):
        preds = []
        for e in fmods:
            if kind == 'clf':
                preds.append(e.predict_proba(X)[:, list(e.classes_).index(1)])
            else:
                preds.append(e.predict(X))
        Z[:, j] = np.mean(preds, axis=0)
    if kind == 'clf':
        return task['meta'].predict_proba(Z)[:, 1]
    m = task['meta']
    return Z @ m['coef'] + m['inter']

def make_input(M, age, amh, fsh, lh, gn, plt_, dor, plan, fert):
    row = {'AMH': amh, 'FSH': fsh, 'LH': lh, '(Gn启动)总天数': gn,
           'D_储备减退': 1.0 if dor == '是' else 0.0, '年龄': age, '血小板计数': plt_,
           'D_方案_温和刺激': 0.0, 'D_方案_PPOS': 0.0, 'D_方案_自然周期': 0.0,
           'D_受精_ICSI': 0.0, 'D_受精_补救混合': 0.0}
    if plan == '温和刺激方案':
        row['D_方案_温和刺激'] = 1.0
    elif plan == 'PPOS方案':
        row['D_方案_PPOS'] = 1.0
    elif plan == '自然周期方案':
        row['D_方案_自然周期'] = 1.0
    if fert == 'ICSI':
        row['D_受精_ICSI'] = 1.0
    elif fert == '补救/混合受精':
        row['D_受精_补救混合'] = 1.0
    return np.array([[row[c] for c in M['features']]], dtype=float)

# ---------------- SHAP 个体解释 ----------------
@st.cache_data(show_spinner=False)
def shap_values(_M, task_name, x_tuple):
    import shap
    M = _M
    task = M['tasks'][task_name]
    X = np.array([x_tuple], dtype=float)
    bg = M['shap_background']
    def f(Z):
        return stack_predict(task, Z)
    explainer = shap.KernelExplainer(f, bg)
    sv = explainer.shap_values(X, nsamples=60, silent=True)
    return np.array(sv).flatten(), explainer.expected_value

def shap_bar_group(pairs, title):
    """pairs: [(变量组名, 组内SHAP值之和)], 按|贡献|排序横条图"""
    pairs = sorted(pairs, key=lambda t: -abs(t[1]))
    labels = [p[0] for p in pairs][::-1]
    vals = [p[1] for p in pairs][::-1]
    colors = ['#C0392B' if v > 0 else '#2E5E8C' for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation='h',
                           marker_color=colors,
                           text=[f'{v:+.2f}' for v in vals], textposition='outside'))
    fig.update_layout(title=dict(text=title, font_size=15),
                      xaxis_title='SHAP值（对该患者预测的贡献）',
                      height=max(200, 130 + 45 * len(pairs)),
                      margin=dict(l=10, r=30, t=40, b=10),
                      plot_bgcolor='white')
    fig.update_xaxes(zeroline=True, zerolinecolor='#999')
    return fig

# ---------------- 界面 ----------------
def main():
    try:
        meta = load_meta()
    except Exception:
        meta = None
    if meta is None:
        # meta.json 缺失时退化为同步加载模型取元信息
        try:
            M0 = get_model()
            meta = {'medians': M0['medians'], 'iqrs': M0['iqrs'], 'version': M0['version'],
                    'cohort_n': M0['cohort_n'], 'trained_at': M0['trained_at']}
        except Exception as e:
            st.error(f'模型文件加载失败: {e}')
            st.stop()

    # 后台预加载模型: 页面立即渲染, 用户填写参数期间模型同步加载
    if not st.session_state.get('_model_warming'):
        st.session_state['_model_warming'] = True
        threading.Thread(target=get_model, daemon=True).start()

    # ---------- 侧边栏输入 ----------
    with st.sidebar:
        st.header('患者参数输入')
        med = meta['medians']
        age = st.number_input('年龄（岁）', 20.0, 45.0, float(med['年龄']), 1.0, placeholder='未提供')
        amh = st.number_input('AMH（ng/mL）', 0.0, 20.0, float(med['AMH']), 0.1, format='%.2f', placeholder='未提供')
        fsh = st.number_input('FSH（IU/L）', 0.5, 40.0, float(med['FSH']), 0.1, format='%.2f', placeholder='未提供')
        lh = st.number_input('LH（IU/L）', 0.5, 40.0, float(med['LH']), 0.1, format='%.2f', placeholder='未提供')
        gn = st.number_input('促性腺激素启动总天数（d）', 1.0, 30.0, float(med['(Gn启动)总天数']), 1.0, placeholder='未提供')
        plt_ = st.number_input('血小板计数（×10⁹/L）', 50.0, 600.0, float(med['血小板计数']), 1.0, placeholder='未提供')
        dor = st.radio('卵巢储备减退', ['否', '是', '未提供'], horizontal=True)
        plan = st.selectbox('促排方案', ['拮抗剂方案', '温和刺激方案', 'PPOS方案', '自然周期方案', '未提供'])
        fert = st.selectbox('受精方式', ['常规IVF', 'ICSI', '补救/混合受精', '未提供'])
        run = st.button('🔮 开始预测', type='primary', use_container_width=True)

    if not run:
        return

    # ---------- 缺失项按队列典型值代入 ----------
    provided = {'年龄': age is not None, 'AMH': amh is not None, 'FSH': fsh is not None,
                'LH': lh is not None, '促性腺激素天数': gn is not None, '血小板计数': plt_ is not None,
                '卵巢储备减退': dor != '未提供', '促排方案': plan != '未提供', '受精方式': fert != '未提供'}
    if age is None:
        age = float(med['年龄'])
    if amh is None:
        amh = float(med['AMH'])
    if fsh is None:
        fsh = float(med['FSH'])
    if lh is None:
        lh = float(med['LH'])
    if gn is None:
        gn = float(med['(Gn启动)总天数'])
    if plt_ is None:
        plt_ = float(med['血小板计数'])
    if dor == '未提供':
        dor = '否'
    if plan == '未提供':
        plan = '拮抗剂方案'
    if fert == '未提供':
        fert = '常规IVF'
    # 模型通常在页面打开时已由后台线程加载完毕; 若用户极快点击则在此等待剩余时间
    try:
        M = get_model()
    except Exception as e:
        st.error(f'模型文件加载失败: {e}')
        st.stop()
    X = make_input(M, age, amh, fsh, lh, gn, plt_, dor, plan, fert)
    T = M['tasks']
    with st.spinner('模型预测中…'):
        pr_reg = float(stack_predict(T['获卵数回归'], X)[0])
        pr_A = float(stack_predict(T['分箱A'], X)[0])
        pr_B = float(stack_predict(T['分箱B'], X)[0])
        pr_L = float(stack_predict(T['LOR分类'], X)[0])
        pr_H = float(stack_predict(T['HOR分类'], X)[0])
    bA = T['分箱A']['bounds']; bB = T['分箱B']['bounds']
    binA = int(np.clip(np.digitize(pr_A, bA), 0, 6))
    binB = int(np.clip(np.digitize(pr_B, bB), 0, 5))
    q = T['获卵数回归']['resid_q']
    lo, hi = pr_reg + q[0], pr_reg + q[4]
    tL, tH = T['LOR分类']['threshold'], T['HOR分类']['threshold']

    # ---------- 结果卡片 ----------
    st.subheader('预测结果')
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('预测获卵数', f'{pr_reg:.1f} 枚', f'90%区间 {max(lo,0):.1f}–{hi:.1f} 枚')
    c2.metric('分箱A档级（7档）', f'第{binA+1}档', BIN_A_LBL[binA])
    c3.metric('分箱B档级（6档）', f'第{binB+1}档', BIN_B_LBL[binB])
    c4.metric('低反应风险（<4枚）', f'{pr_L*100:.1f}%', '高风险' if pr_L >= tL else '低风险',
              delta_color='inverse' if pr_L >= tL else 'normal')
    c5.metric('高反应风险（>20枚）', f'{pr_H*100:.1f}%', '高风险' if pr_H >= tH else '低风险',
              delta_color='inverse' if pr_H >= tH else 'normal')

    # ---------- 个体特征贡献（仅展示使用者实际提供的指标） ----------
    st.subheader('个体特征贡献（SHAP）')
    FEAT_GROUPS = [('年龄', ['年龄']), ('AMH', ['AMH']), ('FSH', ['FSH']), ('LH', ['LH']),
                   ('促性腺激素天数', ['(Gn启动)总天数']), ('血小板计数', ['血小板计数']),
                   ('卵巢储备减退', ['D_储备减退']),
                   ('促排方案', ['D_方案_温和刺激', 'D_方案_PPOS', 'D_方案_自然周期']),
                   ('受精方式', ['D_受精_ICSI', 'D_受精_补救混合'])]
    tabs = st.tabs(['获卵数回归', '分箱A', '分箱B', '低反应分类', '高反应分类'])
    x_tuple = tuple(X.flatten().tolist())
    for tab, tname, ttl in [(tabs[0], '获卵数回归', '各特征对获卵数预测值的贡献（枚）'),
                            (tabs[1], '分箱A', '各特征对分箱A档级评分的贡献'),
                            (tabs[2], '分箱B', '各特征对分箱B档级评分的贡献'),
                            (tabs[3], 'LOR分类', '各特征对低反应预测的贡献（logit）'),
                            (tabs[4], 'HOR分类', '各特征对高反应预测的贡献（logit）')]:
        with tab:
            pairs = []
            for gname, cols in FEAT_GROUPS:
                if provided[gname]:
                    idx = [M['features'].index(c) for c in cols]
                    pairs.append((gname, idx))
            if not pairs:
                continue
            with st.spinner('SHAP计算中（约10秒）…'):
                sv, ev = shap_values(M, tname, x_tuple)
            gpairs = [(gname, float(sum(sv[i] for i in idx))) for gname, idx in pairs]
            st.plotly_chart(shap_bar_group(gpairs, ttl), use_container_width=True)
            st.caption('红色=推高预测，蓝色=拉低预测。')

if __name__ == '__main__':
    main()
