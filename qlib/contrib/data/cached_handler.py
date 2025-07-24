#!/usr/bin/env python3
"""
缓存数据处理器模块
支持在 workflow config 中使用的缓存机制
"""

import os
import pickle
import hashlib
from qlib.contrib.data.handler import Alpha158, Alpha360
from qlib.data.dataset.handler import DataHandlerLP


class CachedDataHandlerMixin:
    """缓存数据处理器混入类"""
    
    def __init__(self, *args, cache_dir='~/.qlib/cache/handler/', enable_cache=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_cache = enable_cache
        self.cache_dir = os.path.expanduser(cache_dir)
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            # 延迟生成缓存键，直到 setup_data 调用时

    def _generate_cache_key(self):
        """生成唯一缓存键：基于class名 + 时间范围 + instruments + processors"""
        # 构建缓存键的组成部分，使用 getattr 来安全获取属性
        key_components = [
            self.__class__.__name__,
            str(getattr(self, 'start_time', 'unknown')),
            str(getattr(self, 'end_time', 'unknown')),
            str(getattr(self, 'fit_start_time', 'unknown')),
            str(getattr(self, 'fit_end_time', 'unknown')),
            str(sorted(getattr(self, 'instruments', [])) if hasattr(self, 'instruments') and getattr(self, 'instruments', None) else 'all'),
            str(getattr(self, 'infer_processors', [])),
            str(getattr(self, 'learn_processors', [])),
        ]
        
        key_str = '_'.join(key_components)
        return hashlib.md5(key_str.encode()).hexdigest() + '.pkl'

    def setup_data(self, *args, **kwargs):
        """重写数据设置方法，添加缓存逻辑"""
        # 确保 enable_cache 属性存在（处理多重继承顺序问题）
        if not hasattr(self, 'enable_cache'):
            self.enable_cache = True
            self.cache_dir = os.path.expanduser('~/.qlib/cache/handler/')
            os.makedirs(self.cache_dir, exist_ok=True)
            
        if not self.enable_cache:
            return super().setup_data(*args, **kwargs)
            
        # 在这里生成缓存键，确保所有属性都已设置
        if not hasattr(self, 'cache_key'):
            self.cache_key = self._generate_cache_key()
            
        cache_path = os.path.join(self.cache_dir, self.cache_key)
        
        if os.path.exists(cache_path):
            print(f"🔄 从缓存加载数据: {os.path.basename(cache_path)}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # 恢复缓存的数据
                self._data = cached_data.get('_data')
                if hasattr(self, '_infer'):
                    self._infer = cached_data.get('_infer')
                if hasattr(self, '_learn'):
                    self._learn = cached_data.get('_learn')
                
                print(f"✅ 缓存加载成功，数据形状: {self._data.shape if self._data is not None else 'None'}")
                return
                
            except Exception as e:
                print(f"⚠️  缓存加载失败，重新计算: {str(e)}")
        
        # 缓存不存在或加载失败，重新计算
        print(f"🔧 重新计算数据并缓存到: {os.path.basename(cache_path)}")
        super().setup_data(*args, **kwargs)
        
        # 保存到缓存
        try:
            cache_data = {'_data': self._data}
            if hasattr(self, '_infer'):
                cache_data['_infer'] = self._infer
            if hasattr(self, '_learn'):
                cache_data['_learn'] = self._learn
                
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 数据已缓存，文件大小: {os.path.getsize(cache_path) / 1024 / 1024:.1f}MB")
            
        except Exception as e:
            print(f"⚠️  缓存保存失败: {str(e)}")


class CachedAlpha158(CachedDataHandlerMixin, Alpha158):
    """带缓存的 Alpha158 数据处理器"""
    
    def __init__(self, *args, cache_dir='~/.qlib/cache/handler/', enable_cache=True, **kwargs):
        # 显式提取缓存相关参数，确保它们被正确传递
        self._cache_dir = cache_dir
        self._enable_cache = enable_cache
        super().__init__(*args, cache_dir=cache_dir, enable_cache=enable_cache, **kwargs)


class CachedAlpha360(CachedDataHandlerMixin, Alpha360):
    """带缓存的 Alpha360 数据处理器"""
    
    def __init__(self, *args, cache_dir='~/.qlib/cache/handler/', enable_cache=True, **kwargs):
        self._cache_dir = cache_dir
        self._enable_cache = enable_cache
        super().__init__(*args, cache_dir=cache_dir, enable_cache=enable_cache, **kwargs)


class CachedDataHandlerLP(CachedDataHandlerMixin, DataHandlerLP):
    """带缓存的通用数据处理器"""
    
    def __init__(self, *args, cache_dir='~/.qlib/cache/handler/', enable_cache=True, **kwargs):
        self._cache_dir = cache_dir
        self._enable_cache = enable_cache
        super().__init__(*args, cache_dir=cache_dir, enable_cache=enable_cache, **kwargs)


# 为了向后兼容，提供别名
CachedDataHandler = CachedDataHandlerLP 