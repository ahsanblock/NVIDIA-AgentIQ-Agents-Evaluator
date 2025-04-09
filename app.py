import streamlit as st
import subprocess
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from glob import glob
from datetime import datetime
from base64 import b64encode

# NVIDIA favicon (you can replace this with the actual NVIDIA favicon)
NVIDIA_FAVICON = "https://www.nvidia.com/favicon.ico"

st.set_page_config(
    page_title="NVIDIA AgentIQ Agents Evaluator",
    page_icon=NVIDIA_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ahsanblock/NVIDIA-AgentIQ-Agents-Evaluator',
        'Report a bug': 'https://github.com/ahsanblock/NVIDIA-AgentIQ-Agents-Evaluator/issues',
        'About': '''
        # NVIDIA AgentIQ - Agents, RAG, LLM Evaluator
        
        A comprehensive evaluation platform for AI Agents, RAG, and LLMs powered by NVIDIA's AgentIQ framework.
        
        Version: 1.0.0
        GitHub: https://github.com/ahsanblock/NVIDIA-AgentIQ-Agents-Evaluator
        '''
    }
)

# Constants and Configurations
MODELS = [
    # Meta Models
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    
    # Mistral Models
    "mistralai/mixtral-8x7b-instruct-v0.1",
    
    # Microsoft Models
    "microsoft/phi-3-mini-4k-instruct",
    "microsoft/phi-3-medium-4k-instruct",
    
    # DeepSeek Models
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    
    # Qwen Models
    "qwen/qwen1.5-72b-chat",
    "qwen/qwen1.5-14b-chat",
]

# Add model grouping information for better organization in UI
MODEL_GROUPS = {
    "Meta": ["meta/llama-3.1-8b-instruct", "meta/llama-3.1-70b-instruct"],
    "Mistral": ["mistralai/mixtral-8x7b-instruct-v0.1"],
    "Microsoft": ["microsoft/phi-3-mini-4k-instruct", "microsoft/phi-3-medium-4k-instruct"],
    "DeepSeek": ["deepseek/deepseek-coder-6.7b-instruct"],
    "Qwen": ["qwen/qwen2-7b-instruct"]
}

NVIDIA_MODEL_MAPPINGS = {
    # Meta Models
    "meta/llama-3.1-8b-instruct": "meta-llama/Llama-2-8b-chat-hf",
    "meta/llama-3.1-70b-instruct": "meta-llama/Llama-2-70b-chat-hf",
    
    # Mistral Models
    "mistralai/mixtral-8x7b-instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    
    # Microsoft Models
    "microsoft/phi-3-mini-4k-instruct": "microsoft/phi-3-mini-4k-instruct",
    "microsoft/phi-3-medium-4k-instruct": "microsoft/phi-3-medium-4k-instruct",
    
    # DeepSeek Models - Using correct format
    "deepseek/deepseek-coder-6.7b-instruct": "deepseek/deepseek-coder-6.7b-instruct",
    
    # Qwen Models
    "qwen/qwen2-7b-instruct": "qwen/qwen2-7b-instruct"
}

def get_available_datasets():
    """Get list of CSV files in data directory"""
    datasets = glob("data/*.csv")
    return [os.path.basename(d) for d in datasets]

def get_nvidia_model_name(model_id):
    """Convert our model ID to NVIDIA API model name"""
    if model_id in NVIDIA_MODEL_MAPPINGS:
        return NVIDIA_MODEL_MAPPINGS[model_id]
    return model_id

def run_evaluation(model, dataset):
    """Run evaluation script and stream output"""
    nvidia_model = get_nvidia_model_name(model)
    cmd = ["bash", "scripts/run_evals.sh", "--model", nvidia_model, "--dataset", f"data/{dataset}"]
    
    # Create a fixed-height scrollable container for logs
    st.markdown("""
        <style>
        .log-container {
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            padding: 1rem;
            border-radius: 8px;
            height: 400px;
            overflow-y: auto;
            margin: 1rem 0;
            font-size: 0.9rem;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .log-container::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        .log-container::-webkit-scrollbar-track {
            background: #2d2d2d;
            border-radius: 4px;
        }
        
        .log-container::-webkit-scrollbar-thumb {
            background: #666;
            border-radius: 4px;
        }
        
        .log-container::-webkit-scrollbar-thumb:hover {
            background: #888;
        }
        
        .nvidia-model-info {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            display: inline-block;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display NVIDIA model info with better styling
    st.markdown(
        f'<div class="nvidia-model-info">🔄 Using NVIDIA model: {nvidia_model}</div>',
        unsafe_allow_html=True
    )
    
    # Create a container for logs with auto-scroll
    log_container = st.empty()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=os.environ.copy()
    )

    # Stream output in real-time with auto-scroll
    full_output = []
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            full_output.append(output.strip())
            # Update the log container with all output
            log_container.markdown(
                f'<div class="log-container">{chr(10).join(full_output)}</div>',
                unsafe_allow_html=True
            )
            
    return process.poll()

def load_metrics(model_id):
    """Load metrics.json and results.json for a model"""
    model_simple = model_id.split('/')[-1]
    model_dir = f"results/{model_simple}"
    
    metrics = {}
    
    # Load metrics.json
    metrics_file = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics.update(json.load(f))
    
    # Calculate token usage from results.json
    results_file = os.path.join(model_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
            prompt_tokens = []
            completion_tokens = []
            
            for record in results:
                for step in record.get("intermediate_steps", []):
                    usage = step.get("payload", {}).get("usage_info", {}).get("token_usage")
                    if usage:
                        prompt_tokens.append(usage.get("prompt_tokens", 0))
                        completion_tokens.append(usage.get("completion_tokens", 0))
            
            if prompt_tokens:
                metrics['average_prompt_tokens'] = sum(prompt_tokens) / len(prompt_tokens)
            if completion_tokens:
                metrics['average_completion_tokens'] = sum(completion_tokens) / len(completion_tokens)
    
    # Load RAG metrics without debug logging
    rag_metrics = {}
    for metric in ['accuracy', 'groundedness', 'relevance']:
        rag_file = f'results/{model_simple}/rag_{metric}_output.json'
        if os.path.exists(rag_file):
            with open(rag_file) as f:
                try:
                    data = json.load(f)
                    
                    # Check for average_score first
                    if data.get('average_score') is not None:
                        rag_metrics[f'rag_{metric}'] = float(data['average_score'])
                        continue
                        
                    # Process eval_output_items if available
                    if 'eval_output_items' in data:
                        valid_scores = []
                        for item in data['eval_output_items']:
                            if item.get('score') is not None:
                                try:
                                    score = float(item['score'])
                                    valid_scores.append(score)
                                except (ValueError, TypeError):
                                    continue
                        
                        if valid_scores:
                            rag_metrics[f'rag_{metric}'] = sum(valid_scores) / len(valid_scores)
                            
                    # Check if data itself is a score
                    elif isinstance(data.get('score'), (int, float)):
                        rag_metrics[f'rag_{metric}'] = float(data['score'])
                        
                except Exception as e:
                    continue
    
    metrics.update(rag_metrics)
    return metrics

def show_comparison_charts(all_metrics):
    """Show comparison charts for all models"""
    
    # Create metrics dataframe with default values
    metrics_df = pd.DataFrame([
        {
            'model': model.split('/')[-1],
            'accuracy': metrics.get('accuracy', 0),
            'latency': metrics.get('average_latency', 0),
            'correct': metrics.get('correct_predictions', 0),
            'total': metrics.get('total_examples', 0),
            'rag_accuracy': metrics.get('rag_accuracy', 0),
            'rag_groundedness': metrics.get('rag_groundedness', 0),
            'rag_relevance': metrics.get('rag_relevance', 0),
            'prompt_tokens': metrics.get('average_prompt_tokens', 0),
            'completion_tokens': metrics.get('average_completion_tokens', 0)
        }
        for model, metrics in all_metrics.items()
    ])

    # Model Performance Section
    st.markdown("""
        ### 📊 Comparison Overview
        Here's how each model performs across different metrics. Let's break down what each chart means:
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            #### Accuracy
            This chart shows the percentage of correct predictions for each model. Higher bars indicate better performance.
            - 100% means all predictions were correct
            - 0% means no predictions were correct
        """)
        fig = px.bar(
            metrics_df,
            x='model',
            y='accuracy',
            title='Accuracy by Model',
            text=metrics_df['accuracy'].apply(lambda x: f'{x:.2%}')
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
            #### Response Time
            This chart shows the average time (in seconds) each model takes to process a request.
            - Lower bars indicate faster response times
            - Higher bars indicate slower response times
        """)
        fig = px.bar(
            metrics_df,
            x='model',
            y='latency',
            title='Average Latency by Model',
            text=metrics_df['latency'].apply(lambda x: f'{x:.2f}s')
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
            #### Prediction Breakdown
            This stacked bar chart shows the number of correct vs incorrect predictions.
            - Green: Number of correct predictions
            - Red: Number of incorrect predictions
            - Total height: Total number of predictions made
        """)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Correct',
            x=metrics_df['model'],
            y=metrics_df['correct'],
            text=metrics_df['correct'],
            textposition='auto',
            marker_color='#2ecc71'
        ))
        fig.add_trace(go.Bar(
            name='Incorrect',
            x=metrics_df['model'],
            y=metrics_df['total'] - metrics_df['correct'],
            text=metrics_df['total'] - metrics_df['correct'],
            textposition='auto',
            marker_color='#e74c3c'
        ))
        fig.update_layout(
            barmode='stack',
            title='Correct vs Total Predictions',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("""
            #### Token Usage Analysis
            This chart compares the average number of tokens used by each model.
            - Prompt Tokens: Number of tokens in the input text
            - Completion Tokens: Number of tokens in the model's response
            - Higher numbers indicate more token usage (and potentially higher costs)
        """)
        token_df = metrics_df[metrics_df['prompt_tokens'] > 0].copy()
        if not token_df.empty:
            token_df = pd.melt(
                token_df,
                id_vars=['model'],
                value_vars=['prompt_tokens', 'completion_tokens'],
                var_name='Type',
                value_name='Tokens'
            )
            token_df['Type'] = token_df['Type'].map({
                'prompt_tokens': 'Prompt',
                'completion_tokens': 'Completion'
            })
            
            fig = px.bar(
                token_df,
                x='model',
                y='Tokens',
                color='Type',
                title='Average Token Usage by Model',
                text='Tokens',
                barmode='group'
            )
            fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No token usage data available for the selected models")

    # RAG Metrics Section
    st.markdown("""
        ### 🎯 RAG (Retrieval-Augmented Generation) Metrics
        RAG metrics help us understand how well the model uses external information in its responses:
    """)
    col5, col6, col7 = st.columns(3)

    rag_colors = {
        'rag_accuracy': '#3498db',
        'rag_groundedness': '#2ecc71',
        'rag_relevance': '#9b59b6'
    }

    with col5:
        st.markdown("""
            #### RAG Accuracy
            Measures how accurate the model's responses are when using retrieved information.
            - Score ranges from 0 to 1
            - Higher scores indicate better accuracy
        """)
        fig = px.bar(
            metrics_df,
            x='model',
            y='rag_accuracy',
            title='RAG Accuracy',
            text=metrics_df['rag_accuracy'].apply(lambda x: f'{x:.3f}'),
            color_discrete_sequence=[rag_colors['rag_accuracy']]
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            yaxis_title='Score',
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.markdown("""
            #### RAG Groundedness
            Indicates how well the model's responses are grounded in the retrieved information.
            - Score ranges from 0 to 1
            - Higher scores mean better use of source information
        """)
        fig = px.bar(
            metrics_df,
            x='model',
            y='rag_groundedness',
            title='RAG Groundedness',
            text=metrics_df['rag_groundedness'].apply(lambda x: f'{x:.3f}'),
            color_discrete_sequence=[rag_colors['rag_groundedness']]
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            yaxis_title='Score',
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col7:
        st.markdown("""
            #### RAG Relevance
            Shows how relevant the model's responses are to the input queries.
            - Score ranges from 0 to 1
            - Higher scores indicate more relevant responses
        """)
        fig = px.bar(
            metrics_df,
            x='model',
            y='rag_relevance',
            title='RAG Relevance',
            text=metrics_df['rag_relevance'].apply(lambda x: f'{x:.3f}'),
            color_discrete_sequence=[rag_colors['rag_relevance']]
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            yaxis_title='Score',
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

def load_profiling_metrics(model_id):
    """Load and process profiling metrics"""
    model_simple = model_id.split('/')[-1]
    profiling_file = f"results/{model_simple}/workflow_profiling_metrics.json"
    if not os.path.exists(profiling_file):
        return None
        
    with open(profiling_file) as f:
        return json.load(f)

def load_inference_optimization(model_id):
    """Load inference optimization metrics"""
    model_simple = model_id.split('/')[-1]
    optimization_file = f"results/{model_simple}/inference_optimization.json"
    if not os.path.exists(optimization_file):
        return None
        
    with open(optimization_file) as f:
        return json.load(f)

def show_detailed_analysis(model_name):
    """Show detailed analysis for a specific agent"""
    st.markdown(f"### 🔍 Detailed Analysis: {model_name}")
    st.markdown("""
    In-depth analysis of agent performance including RAG capabilities, 
    workflow efficiency, and decision-making metrics.
    """)
    
    model_id = model_name
    results_file = f'results/{model_id}/results.json'
    
    if not os.path.exists(results_file):
        st.error(f"No results found for {model_name}")
        return
        
    with open(results_file) as f:
        results = json.load(f)

    # Create tabs for different types of analysis
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Prediction Analysis",
        "⚡ Performance Metrics",
        "🔄 Workflow Analysis",
        "📊 Advanced Analytics"
    ])
    
    with tab1:
        st.markdown("""
            ### Prediction Analysis
            This section shows detailed analysis of the model's predictions:
            - Summary metrics of overall performance
            - Distribution of response lengths
            - Confusion matrix showing prediction patterns
            - Detailed breakdown of individual predictions
        """)
        predictions_data = []
        for item in results:
            if 'body' in item and 'generated_answer' in item and 'label' in item:
                prediction = 'phish' if any(w in item['generated_answer'].lower() 
                                          for w in ['phish', 'scam', 'suspicious', 'fraud']) else 'benign'
                predictions_data.append({
                    'Email': item['body'][:100] + '...',
                    'Actual': item['label'],
                    'Predicted': prediction,
                    'Correct': '✅' if prediction == item['label'] else '❌',
                    'Model Response': item['generated_answer'][:200] + '...',
                    'Response Length': len(item['generated_answer'])
                })
        
        if predictions_data:
            df = pd.DataFrame(predictions_data)
            
            # Summary metrics
            st.markdown("#### Key Performance Indicators")
            col1, col2, col3, col4 = st.columns(4)
            correct_count = df['Correct'].value_counts().get('✅', 0)
            total_count = len(df)
            accuracy = correct_count / total_count if total_count > 0 else 0
            avg_response_length = df['Response Length'].mean()
            
            col1.metric("Total Predictions", total_count, help="Total number of predictions made by the model")
            col2.metric("Correct Predictions", correct_count, help="Number of correct predictions")
            col3.metric("Accuracy", f"{accuracy:.2%}", help="Percentage of correct predictions")
            col4.metric("Avg Response Length", f"{avg_response_length:.0f}", help="Average length of model responses in characters")
            
            # Response Length Distribution
            st.markdown("""
                #### Response Length Distribution
                This histogram shows the distribution of response lengths:
                - X-axis: Length of response in characters
                - Y-axis: Number of responses
                - Colors indicate correct vs incorrect predictions
            """)
            fig = px.histogram(
                df,
                x='Response Length',
                nbins=30,
                title='Distribution of Response Lengths',
                color='Correct'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion Matrix
            st.markdown("""
                #### Confusion Matrix
                This matrix shows the relationship between actual and predicted labels:
                - Rows: Actual labels
                - Columns: Predicted labels
                - Numbers: Count of predictions in each category
                - Darker colors indicate higher counts
            """)
            confusion_data = pd.crosstab(df['Actual'], df['Predicted'])
            fig = px.imshow(
                confusion_data,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Predictions Table
            st.markdown("""
                #### Detailed Predictions
                This table shows individual predictions made by the model:
                - Email Content: First 100 characters of the input email
                - Actual Label: True classification of the email
                - Predicted Label: Model's prediction
                - Correct?: Whether the prediction was correct
                - Model Response: First 200 characters of the model's response
            """)
            st.dataframe(
                df,
                height=400,
                column_config={
                    "Email": st.column_config.TextColumn("Email Content", width="medium"),
                    "Actual": st.column_config.TextColumn("Actual Label", width="small"),
                    "Predicted": st.column_config.TextColumn("Predicted Label", width="small"),
                    "Correct": st.column_config.TextColumn("Correct?", width="small"),
                    "Model Response": st.column_config.TextColumn("Model Response", width="large"),
                }
            )
    
    with tab2:
        st.markdown("""
            ### Performance Metrics
            This section provides detailed performance analysis:
            - Response time distribution
            - Memory usage metrics
            - Inference optimization metrics
            - Batch processing efficiency
        """)
        
        # Load profiling metrics
        profiling_metrics = load_profiling_metrics(model_id)
        if profiling_metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    #### Response Time Distribution
                    This histogram shows how response times are distributed:
                    - X-axis: Response time in milliseconds
                    - Y-axis: Number of responses
                    - Pattern indicates processing speed consistency
                """)
                response_times = [step.get('duration_ms', 0) for step in profiling_metrics.get('steps', [])]
                fig = px.histogram(
                    x=response_times,
                    nbins=20,
                    title='Response Time Distribution (ms)',
                    labels={'x': 'Response Time (ms)', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                    #### Memory Usage
                    This chart shows key memory metrics:
                    - Peak Memory: Maximum memory used
                    - Average Memory: Typical memory usage
                    - Memory Growth: Increase in memory over time
                """)
                if 'memory_metrics' in profiling_metrics:
                    memory_data = profiling_metrics['memory_metrics']
                    memory_df = pd.DataFrame([
                        {"Metric": "Peak Memory", "GB": memory_data.get('peak_memory_gb', 0)},
                        {"Metric": "Average Memory", "GB": memory_data.get('average_memory_gb', 0)},
                        {"Metric": "Memory Growth", "GB": memory_data.get('memory_growth_gb', 0)}
                    ])
                    fig = px.bar(
                        memory_df,
                        x='Metric',
                        y='GB',
                        title='Memory Usage Metrics',
                        text='GB'
                    )
                    fig.update_traces(texttemplate='%{text:.2f} GB')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance metrics available for this model. Run an evaluation to generate performance data.")
        
        # Load inference optimization metrics
        optimization_metrics = load_inference_optimization(model_id)
        if optimization_metrics:
            st.markdown("""
                ### Inference Optimization Metrics
                These metrics show how efficiently the model processes requests:
            """)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    #### Throughput
                    This gauge shows the processing speed:
                    - Value: Current tokens processed per second
                    - Maximum: Peak tokens per second achieved
                """)
                throughput_data = optimization_metrics.get('throughput_metrics', {})
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=throughput_data.get('tokens_per_second', 0),
                    title={'text': "Tokens per Second"},
                    gauge={'axis': {'range': [None, throughput_data.get('peak_tokens_per_second', 100)]}}
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                    #### Batch Processing
                    This chart shows batch processing efficiency:
                    - Avg Batch Size: Average number of items processed together
                    - Batch Utilization: How effectively the batching is used (%)
                """)
                batch_metrics = optimization_metrics.get('batch_processing', {})
                efficiency_df = pd.DataFrame([
                    {"Metric": "Avg Batch Size", "Value": batch_metrics.get('average_batch_size', 0)},
                    {"Metric": "Batch Utilization", "Value": batch_metrics.get('batch_utilization_percent', 0)}
                ])
                fig = px.bar(
                    efficiency_df,
                    x='Metric',
                    y='Value',
                    title='Batch Processing Efficiency',
                    text='Value'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No optimization metrics available for this model. Run an evaluation to generate optimization data.")

    with tab3:
        st.subheader("Workflow Analysis")
        
        # Load workflow output
        workflow_file = f'results/{model_id}/workflow_output.json'
        if os.path.exists(workflow_file):
            with open(workflow_file) as f:
                workflow_data = json.load(f)
            
            # Timeline visualization
            events = []
            for item in workflow_data:
                if 'timestamp' in item and 'event_type' in item:
                    events.append({
                        'Event': item['event_type'],
                        'Timestamp': pd.to_datetime(item['timestamp']),
                        'Duration': item.get('duration_ms', 0)
                    })
            
            if events:
                events_df = pd.DataFrame(events)
                events_df = events_df.sort_values('Timestamp')
                
                # Event timeline
                fig = px.timeline(
                    events_df,
                    x_start='Timestamp',
                    y='Event',
                    title='Workflow Event Timeline'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Event distribution
                fig = px.bar(
                    events_df['Event'].value_counts().reset_index(),
                    x='index',
                    y='Event',
                    title='Event Type Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Display Gantt chart if available
        gantt_file = f'results/{model_id}/gantt_chart.png'
        if os.path.exists(gantt_file):
            st.subheader("Workflow Gantt Chart")
            st.image(gantt_file)
    
    with tab4:
        st.subheader("Advanced Analytics")
        
        # Load standardized data if available
        data_file = f'results/{model_id}/standardized_data_all.csv'
        if os.path.exists(data_file):
            df_std = pd.read_csv(data_file)
            
            # Time series analysis
            if 'timestamp' in df_std.columns:
                df_std['timestamp'] = pd.to_datetime(df_std['timestamp'])
                st.subheader("Performance Over Time")
                
                # Rolling average of key metrics
                metrics_to_plot = [col for col in df_std.columns if 'score' in col.lower() or 'accuracy' in col.lower()]
                for metric in metrics_to_plot:
                    fig = px.line(
                        df_std,
                        x='timestamp',
                        y=metric,
                        title=f'{metric} Over Time',
                        line_shape='spline'
                    )
                    fig.add_traces(
                        px.line(
                            df_std.rolling(window=10).mean(),
                            x='timestamp',
                            y=metric,
                            line_shape='spline'
                        ).data
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Error analysis
        if predictions_data:
            df = pd.DataFrame(predictions_data)
            incorrect_cases = df[df['Correct'] == '❌']
            
            if not incorrect_cases.empty:
                st.subheader("Error Analysis")
                
                # Group errors by patterns
                error_patterns = []
                for _, row in incorrect_cases.iterrows():
                    response = row['Model Response'].lower()
                    if 'uncertain' in response or 'unclear' in response:
                        pattern = 'Uncertainty'
                    elif 'no indication' in response or 'cannot determine' in response:
                        pattern = 'Lack of Information'
                    elif 'however' in response or 'but' in response:
                        pattern = 'Conflicting Signals'
                    else:
                        pattern = 'Other'
                    error_patterns.append(pattern)
                
                error_distribution = pd.Series(error_patterns).value_counts()
                fig = px.pie(
                    values=error_distribution.values,
                    names=error_distribution.index,
                    title='Error Pattern Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Sample error cases
                st.subheader("Sample Error Cases")
                st.dataframe(
                    incorrect_cases.sample(min(5, len(incorrect_cases)))[
                        ['Email', 'Actual', 'Predicted', 'Model Response']
                    ]
                )

def main():
    # Add custom CSS for NVIDIA branding
    st.markdown("""
        <style>
        .nvidia-header {
            color: #76b900;
            font-size: 2.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .feature-card {
            background-color: #1e1e1e;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #76b900;
            margin-bottom: 1rem;
        }
        .highlight-text {
            color: #76b900;
            font-weight: 600;
        }
        .dashboard-highlight {
            background-color: #2c3e50;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main Header
    st.markdown('<div class="nvidia-header">NVIDIA AgentIQ - Agents, RAG, LLM Evaluator</div>', unsafe_allow_html=True)
    
    
    
    # Dashboard Highlights
    st.markdown("""
        ## 💫 Dashboard Key Highlights
        
        <div class="dashboard-highlight">
        • <span class="highlight-text">Comprehensive Metrics:</span> Track accuracy, latency, token usage, and RAG performance<br>
        • <span class="highlight-text">Interactive Visualizations:</span> Real-time charts and comparative analysis tools<br>
        • <span class="highlight-text">Workflow Analysis:</span> Detailed agent behavior and decision process tracking<br>
        • <span class="highlight-text">Resource Monitoring:</span> Memory usage, throughput, and optimization metrics
        </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.markdown("## 🛠️ Configuration")
        
        # API Key input with better styling
        st.markdown("### 🔑 NVIDIA API Configuration")
        api_key = st.text_input(
            "Enter your NVIDIA API key",
            value=os.environ.get('NVIDIA_API_KEY', ''),
            type="password",
            help="Required for accessing NVIDIA AI endpoints"
        )
        
        if not api_key:
            st.warning("Please enter your NVIDIA API key to proceed")
        else:
            os.environ['NVIDIA_API_KEY'] = api_key
        
        st.markdown("### 🤖 Agent Selection")
        # Dataset selection with default to small.csv
        datasets = get_available_datasets()
        default_index = next((i for i, d in enumerate(datasets) if d == "small.csv"), 0)
        selected_dataset = st.selectbox(
            "Select Dataset",
            datasets,
            index=default_index,
            help="Choose a dataset for evaluation"
        )

        # Get list of models with existing results
        existing_results = set(d for d in os.listdir("results") 
                           if os.path.isdir(os.path.join("results", d)) 
                           and os.path.exists(os.path.join("results", d, "metrics.json")))
        
        st.markdown('<p class="section-header">Model Selection</p>', unsafe_allow_html=True)
        
        # Create a dictionary of model selections with visual indicators, grouped by provider
        model_selections = {}
        for group, models in MODEL_GROUPS.items():
            with st.container():
                st.markdown(f'<div class="model-group">', unsafe_allow_html=True)
                st.markdown(f"#### {group}")
                for model in models:
                    model_simple = model.split('/')[-1]
                    has_results = model_simple in existing_results
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        model_selections[model] = st.checkbox(
                            model,
                            value=has_results,
                            help="✅ Results available" if has_results else "⚠️ Needs evaluation"
                        )
                    with col2:
                        if has_results:
                            st.markdown('<span class="success-indicator">✓</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="warning-indicator">⚠</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        selected_models = [model for model, selected in model_selections.items() if selected]
        
        # Determine which models need evaluation
        models_to_evaluate = [model for model in selected_models 
                            if model.split('/')[-1] not in existing_results]
        
        if models_to_evaluate:
            st.markdown('<p class="section-header">Models to Evaluate</p>', unsafe_allow_html=True)
            for model in models_to_evaluate:
                st.markdown(
                    f'<div style="color: #e0e0e0; font-size: 0.85rem; padding: 0.2rem 0;">• {model.split("/")[-1]}</div>',
                    unsafe_allow_html=True
                )
            
            if st.button("🚀 Run Evaluation", type="primary", disabled=not api_key):
                st.markdown('<p class="section-header">Evaluation Progress</p>', unsafe_allow_html=True)
                progress_bar = st.progress(0)
                
                eval_container = st.container()
                with eval_container:
                    for idx, model in enumerate(models_to_evaluate):
                        st.markdown(
                            f'<div style="color: #e0e0e0; font-size: 0.9rem; margin: 0.5rem 0;">Running {model}</div>',
                            unsafe_allow_html=True
                        )
                        exit_code = run_evaluation(model, selected_dataset)
                        if exit_code != 0:
                            st.error(f"❌ Evaluation failed for {model}")
                        else:
                            st.success(f"✅ Evaluation completed for {model}")
                        progress_bar.progress((idx + 1) / len(models_to_evaluate))
                    
                st.success("🎉 All Evaluations Complete!")
                st.info("💡 Results will appear in comparison view")

    # Main content area with modern tabs
    tab1, tab2 = st.tabs(["📊 Comparison", "🔍 Detailed Analysis"])
    
    with tab1:
        if not selected_models:
            st.info("👈 Please select models to compare in the sidebar.")
        else:
            all_metrics = {}
            for model in selected_models:
                metrics = load_metrics(model)
                if metrics:
                    all_metrics[model] = metrics

            if all_metrics:
                show_comparison_charts(all_metrics)
            else:
                st.info("⚠️ No evaluation results found for the selected models. Please run evaluations first.")
                
    with tab2:
        if not existing_results:
            st.info("📝 No models available for detailed analysis. Please run evaluations first.")
        else:
            selected_model = st.selectbox(
                "Select Model for Detailed Analysis",
                list(existing_results)
            )
            show_detailed_analysis(selected_model)

if __name__ == "__main__":
    main()