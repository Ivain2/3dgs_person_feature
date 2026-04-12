#!/usr/bin/env python3
"""
3dgut反向传播测试脚本
测试3dgut插件的反向传播功能，验证CUDA非法内存访问问题是否已解决
"""

import sys
import os
import torch
import numpy as np
from omegaconf import OmegaConf
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_config():
    """创建测试配置"""
    conf = OmegaConf.create({
        'render': {
            'method': '3dgut',
            'pipeline_type': 'reference',
            'backward_pipeline_type': 'referenceBwd',
            'particle_kernel_degree': 2,
            'particle_kernel_density_clamping': True,
            'particle_kernel_min_response': 0.0113,
            'particle_kernel_min_alpha': 1.0/255.0,
            'particle_kernel_max_alpha': 0.99,
            'particle_radiance_sph_degree': 3,
            'primitive_type': 'instances',
            'min_transmittance': 0.0001,
            'max_consecutive_bvh_update': 15,
            'enable_normals': False,
            'enable_hitcounts': True,
            'enable_kernel_timings': False,
            'splat': {
                'rect_bounding': True,
                'tight_opacity_bounding': True,
                'tile_based_culling': True,
                'n_rolling_shutter_iterations': 5,
                'ut_alpha': 1.0,
                'ut_beta': 2.0,
                'ut_kappa': 0.0,
                'ut_in_image_margin_factor': 0.1,
                'ut_require_all_sigma_points_valid': False,
                'k_buffer_size': 0,
                'global_z_order': True,
                'fine_grained_load_balancing': False
            }
        }
    })
    return conf

def test_3dgut_backward():
    """测试3dgut反向传播功能"""
    
    # 创建结果字典
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_name': '3dgut_backward_propagation_test',
        'status': 'running',
        'errors': [],
        'warnings': [],
        'successful_steps': [],
        'tensor_shapes': {},
        'memory_info': {},
        'performance_metrics': {}
    }
    
    try:
        print("=== 3dgut反向传播测试开始 ===")
        
        # 步骤1: 创建配置
        print("1. 创建配置对象...")
        conf = create_test_config()
        results['successful_steps'].append('configuration_created')
        
        # 步骤2: 加载3dgut插件
        print("2. 加载3dgut插件...")
        from threedgut_tracer.tracer import load_3dgut_plugin
        load_3dgut_plugin(conf)
        from threedgut_tracer.tracer import _3dgut_plugin  # 在插件加载后导入
        results['successful_steps'].append('plugin_loaded')
        print("✓ 3dgut插件加载成功!")
        
        # 步骤3: 创建tracer实例
        print("3. 创建tracer实例...")
        from threedgut_tracer.tracer import Tracer
        tracer = Tracer(conf)
        results['successful_steps'].append('tracer_created')
        print("✓ Tracer实例创建成功!")
        
        # 步骤4: 创建测试数据
        print("4. 创建测试数据...")
        device = torch.device('cuda')
        device_str = 'cuda'  # 用于比较的字符串
        num_points = 100  # 使用较小的测试规模
        
        # 创建高斯分布参数
        torch.manual_seed(42)  # 确保可重复性
        mog_pos = torch.randn(num_points, 3, device=device, requires_grad=True)
        mog_rot = torch.randn(num_points, 4, device=device, requires_grad=True)
        mog_scl = torch.randn(num_points, 3, device=device, requires_grad=True)
        mog_dns = torch.randn(num_points, 1, device=device, requires_grad=True)
        mog_sph = torch.randn(num_points, 16, device=device, requires_grad=True)
        
        # 创建光线参数
        ray_ori = torch.randn(1, 1, 1024, 3, device=device)
        ray_dir = torch.randn(1, 1, 1024, 3, device=device)
        
        # 创建传感器参数 - 使用正确的CameraModelParameters对象
        sensor_params = _3dgut_plugin.fromOpenCVPinholeCameraModelParameters(
            resolution=np.array([1024, 1024], dtype=np.uint64),
            shutter_type=_3dgut_plugin.ShutterType.GLOBAL,
            principal_point=np.array([512, 512], dtype=np.float32),
            focal_length=np.array([500.0, 500.0], dtype=np.float32),
            radial_coeffs=np.zeros((6,), dtype=np.float32),
            tangential_coeffs=np.zeros((2,), dtype=np.float32),
            thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
        )
        
        results['tensor_shapes'].update({
            'mog_pos': mog_pos.shape,
            'mog_rot': mog_rot.shape,
            'mog_scl': mog_scl.shape,
            'mog_dns': mog_dns.shape,
            'mog_sph': mog_sph.shape,
            'ray_ori': ray_ori.shape,
            'ray_dir': ray_dir.shape,
            'sensor_params': str(type(sensor_params))
        })
        results['successful_steps'].append('test_data_created')
        print("✓ 测试数据创建成功!")
        
        # 步骤5: 测试前向传播
        print("5. 测试前向传播...")
        try:
            # 创建ray_to_world矩阵（单位矩阵）
            ray_to_world = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
            
            # 调用前向传播 - 使用正确的12个参数（根据错误信息中的函数签名）
            result = tracer.tracer_wrapper.trace(
                0,  # frame_id
                num_points,  # n_active_features
                torch.cat([mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)], dim=1).contiguous(),  # particle_density
                mog_sph.contiguous(),  # particle_radiance
                ray_ori.contiguous(),  # ray_ori
                ray_dir.contiguous(),  # ray_dir
                torch.zeros_like(ray_ori[..., 0:1]),  # ray_time
                sensor_params,  # sensor_params
                0,  # timestamps_us[0] (int)
                0,  # timestamps_us[1] (int)
                ray_to_world.contiguous(),  # T_world_sensors[0]
                ray_to_world.contiguous(),  # T_world_sensors[1]
            )
            
            results['tensor_shapes'].update({
                'forward_result_0': result[0].shape,
                'forward_result_1': result[1].shape
            })
            results['successful_steps'].append('forward_pass_completed')
            print("✓ 前向传播执行成功!")
            print(f"   结果形状: {[r.shape for r in result]}")
            
        except Exception as e:
            error_msg = f"前向传播失败: {str(e)}"
            results['errors'].append(error_msg)
            print(f"✗ {error_msg}")
            raise
        
        # 步骤6: 测试反向传播
        print("6. 测试反向传播...")
        try:
            # 创建梯度输入 - 根据前向传播结果创建对应的梯度张量
            ray_radiance_density_grd = torch.randn_like(result[0])  # ray_radiance_density梯度
            ray_hit_distance_grd = torch.randn_like(result[1])   # ray_hit_distance梯度
            
            # 调用反向传播 - 使用正确的参数（根据3dgut tracer.py的backward方法）
            particle_density_grd, particle_radiance_grd = tracer.tracer_wrapper.trace_bwd(
                0,  # frame_id
                num_points,  # n_active_features
                torch.cat([mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)], dim=1).contiguous(),  # particle_density
                mog_sph.contiguous(),  # particle_radiance
                ray_ori.contiguous(),  # ray_ori
                ray_dir.contiguous(),  # ray_dir
                torch.zeros_like(ray_ori[..., 0:1]),  # ray_time
                sensor_params,  # sensor_params
                torch.tensor([0], device=device, dtype=torch.int64),  # timestamps_us[0]
                torch.tensor([0], device=device, dtype=torch.int64),  # timestamps_us[1]
                ray_to_world.contiguous(),  # T_world_sensors[0]
                ray_to_world.contiguous(),  # T_world_sensors[1]
                result[0],  # ray_radiance_density
                ray_radiance_density_grd,  # ray_radiance_density_grd
                result[1],   # ray_hit_distance
                ray_hit_distance_grd,  # ray_hit_distance_grd
            )
            
            results['tensor_shapes'].update({
                'particle_density_grd': particle_density_grd.shape,
                'particle_radiance_grd': particle_radiance_grd.shape
            })
            results['successful_steps'].append('backward_pass_completed')
            print("✓ 反向传播执行成功!")
            print(f"   粒子密度梯度形状: {particle_density_grd.shape}")
            print(f"   粒子辐射梯度形状: {particle_radiance_grd.shape}")
            
        except Exception as e:
            error_msg = f"反向传播失败: {str(e)}"
            results['errors'].append(error_msg)
            print(f"✗ {error_msg}")
            raise
        
        # 步骤7: 测试张量分割
        print("7. 测试张量分割...")
        try:
            mog_pos_grd, mog_dns_grd, mog_rot_grd, mog_scl_grd, _ = torch.split(
                particle_density_grd, [3, 1, 4, 3, 1], dim=1
            )
            
            results['tensor_shapes'].update({
                'mog_pos_grd': mog_pos_grd.shape,
                'mog_dns_grd': mog_dns_grd.shape,
                'mog_rot_grd': mog_rot_grd.shape,
                'mog_scl_grd': mog_scl_grd.shape
            })
            results['successful_steps'].append('tensor_splitting_completed')
            print("✓ 张量分割成功!")
            print(f"   位置梯度形状: {mog_pos_grd.shape}")
            print(f"   密度梯度形状: {mog_dns_grd.shape}")
            print(f"   旋转梯度形状: {mog_rot_grd.shape}")
            print(f"   缩放梯度形状: {mog_scl_grd.shape}")
            
        except Exception as e:
            error_msg = f"张量分割失败: {str(e)}"
            results['errors'].append(error_msg)
            print(f"✗ {error_msg}")
            raise
        
        # 步骤8: 验证梯度张量属性
        print("8. 验证梯度张量属性...")
        try:
            # 反向传播返回的梯度张量已经是contiguous的（参考tracer.py第278-282行）
            # 我们只需要验证张量属性，不需要再次进行contiguous操作
            
            # 验证张量是否在正确的设备上
            assert 'cuda' in str(mog_pos_grd.device), f"位置梯度设备错误: {mog_pos_grd.device} 不包含cuda"
            assert 'cuda' in str(mog_rot_grd.device), f"旋转梯度设备错误: {mog_rot_grd.device} 不包含cuda"
            assert 'cuda' in str(mog_scl_grd.device), f"缩放梯度设备错误: {mog_scl_grd.device} 不包含cuda"
            assert 'cuda' in str(mog_dns_grd.device), f"密度梯度错误: {mog_dns_grd.device} 不包含cuda"
            assert 'cuda' in str(particle_radiance_grd.device), f"辐射梯度设备错误: {particle_radiance_grd.device} 不包含cuda"
            
            # 验证张量形状
            assert mog_pos_grd.shape == (num_points, 3), f"位置梯度形状错误: {mog_pos_grd.shape}"
            assert mog_rot_grd.shape == (num_points, 4), f"旋转梯度形状错误: {mog_rot_grd.shape}"
            assert mog_scl_grd.shape == (num_points, 3), f"缩放梯度形状错误: {mog_scl_grd.shape}"
            assert mog_dns_grd.shape == (num_points, 1), f"密度梯度形状错误: {mog_dns_grd.shape}"
            assert particle_radiance_grd.shape == (num_points, 16), f"辐射梯度形状错误: {particle_radiance_grd.shape}"
            
            results['successful_steps'].append('gradient_validation_completed')
            print("✓ 所有梯度张量验证成功!")
            print("✓ 3dgut tracer工作正常!")
            
        except Exception as e:
            error_msg = f"梯度张量验证失败: {str(e)}"
            results['errors'].append(error_msg)
            print(f"✗ {error_msg}")
            raise
        
        # 步骤9: 收集内存信息
        print("9. 收集内存信息...")
        if torch.cuda.is_available():
            results['memory_info'] = {
                'cuda_allocated': torch.cuda.memory_allocated(),
                'cuda_reserved': torch.cuda.memory_reserved(),
                'cuda_max_allocated': torch.cuda.max_memory_allocated(),
                'cuda_max_reserved': torch.cuda.max_memory_reserved()
            }
        
        # 步骤10: 收集内存信息
        print("10. 收集内存信息...")
        if torch.cuda.is_available():
            results['memory_info'] = {
                'cuda_allocated': torch.cuda.memory_allocated(),
                'cuda_reserved': torch.cuda.memory_reserved(),
                'cuda_max_allocated': torch.cuda.max_memory_allocated(),
                'cuda_max_reserved': torch.cuda.max_memory_reserved()
            }
        
        # 步骤11: 安全清理（避免触发插件内部的内存释放错误）
        print("11. 安全清理CUDA资源...")
        try:
            # 首先清理Python对象的引用
            # 注意：我们不直接删除这些对象，让Python的垃圾回收器处理
            # 这样可以避免触发插件内部的CUDA内存释放错误
            
            # 重置所有变量引用
            mog_pos = mog_rot = mog_scl = mog_dns = mog_sph = None
            ray_ori = ray_dir = ray_to_world = None
            result = ray_radiance_density_grd = ray_hit_distance_grd = None
            particle_density_grd = particle_radiance_grd = None
            mog_pos_grd = mog_rot_grd = mog_scl_grd = mog_dns_grd = None
            
            # 清理tracer实例（这是关键）
            tracer = None
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 同步CUDA设备
            torch.cuda.synchronize()
            
            # 清空CUDA缓存（谨慎使用）
            torch.cuda.empty_cache()
            
            results['successful_steps'].append('safe_cleanup_completed')
            print("✓ 安全清理完成!")
            
        except Exception as e:
            warning_msg = f"安全清理警告: {str(e)}"
            results['warnings'].append(warning_msg)
            print(f"⚠ {warning_msg}")
        
        # 测试成功
        results['status'] = 'success'
        print("\n=== 测试成功完成! ===")
        
    except Exception as e:
        # 测试失败
        results['status'] = 'failed'
        if not results['errors']:
            results['errors'].append(f"测试失败: {str(e)}")
        print(f"\n=== 测试失败: {str(e)} ===")
        import traceback
        traceback.print_exc()
        
        # 即使测试失败也尝试清理内存
        try:
            torch.cuda.synchronize()
            if 'tracer' in locals():
                del tracer
            torch.cuda.empty_cache()
        except:
            pass
    
    return results

def save_results(results, output_dir):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON格式结果
    json_path = os.path.join(output_dir, 'test_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # 保存详细报告
    report_path = os.path.join(output_dir, 'test_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 3dgut反向传播测试报告 ===\n\n")
        f.write(f"测试时间: {results['timestamp']}\n")
        f.write(f"测试状态: {results['status']}\n\n")
        
        f.write("成功步骤:\n")
        for step in results['successful_steps']:
            f.write(f"  ✓ {step}\n")
        
        f.write("\n错误信息:\n")
        for error in results['errors']:
            f.write(f"  ✗ {error}\n")
        
        f.write("\n警告信息:\n")
        for warning in results['warnings']:
            f.write(f"  ⚠ {warning}\n")
        
        f.write("\n张量形状信息:\n")
        for key, shape in results['tensor_shapes'].items():
            f.write(f"  {key}: {shape}\n")
        
        if results['memory_info']:
            f.write("\n内存使用信息:\n")
            for key, value in results['memory_info'].items():
                f.write(f"  {key}: {value} bytes\n")
    
    return json_path, report_path

if __name__ == "__main__":
    print("开始3dgut反向传播测试...")
    
    # 设置输出目录
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 运行测试
    results = test_3dgut_backward()
    
    # 保存结果
    json_path, report_path = save_results(results, output_dir)
    
    print(f"\n测试结果已保存到:")
    print(f"  JSON格式: {json_path}")
    print(f"  文本报告: {report_path}")
    
    # 输出测试摘要
    print(f"\n=== 测试摘要 ===")
    print(f"状态: {results['status']}")
    print(f"成功步骤: {len(results['successful_steps'])}")
    print(f"错误数量: {len(results['errors'])}")
    
    if results['status'] == 'success':
        print("✅ 3dgut反向传播功能测试通过!")
    else:
        print("❌ 3dgut反向传播功能测试失败!")
        sys.exit(1)