"""Run from the repository root, separately for each framework and responsibility.

python tests/test_regressions.py --library HtFLlib --suite models
python tests/test_regressions.py --library PFLlib --suite state
python tests/test_regressions.py --library HtFLlib --suite data
"""
import argparse
import ast
import contextlib
import copy
import gc
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
selection = argparse.ArgumentParser()
selection.add_argument('--library', choices=['HtFLlib', 'PFLlib'], required=True)
selection.add_argument('--suite', choices=['models', 'state', 'data', 'cli'], required=True)
options, unittest_args = selection.parse_known_args()
torch.set_num_threads(1)

if options.suite == 'data':
    sys.path.insert(0, str(ROOT / options.library / 'dataset'))
    import generate_TinyImagenet as generator
else:
    sys.path.insert(0, str(ROOT / options.library / 'system'))
    import main


def parse_args(arguments):
    """Execute the entry point's original parser declarations without training."""
    tree = ast.parse(Path(main.__file__).read_text(encoding='utf-8'))
    entry = next(node for node in tree.body if isinstance(node, ast.If))
    declarations = []
    for node in entry.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'args' for target in node.targets
        ):
            break
        declarations.append(node)
    namespace = vars(main).copy()
    with patch.object(sys, 'argv', ['main.py', *arguments]):
        exec(compile(ast.Module(body=declarations, type_ignores=[]), main.__file__, 'exec'), namespace)
        return namespace['parser'].parse_args(arguments)


def configure(dataset, classes, extra=()):
    args = parse_args(['-dev', 'cpu', '-data', dataset, '-nb', str(classes), *extra])
    # Capture the real entry point's model construction, excluding file-based training.
    class Capture:
        def __init__(self, supplied, times):
            pass
        def train(self):
            pass
    with patch.object(main, 'FedRE', Capture), patch.object(main, 'average_data', lambda **kw: None), contextlib.redirect_stdout(io.StringIO()):
        main.run(args)
    return args


class ModelTests(unittest.TestCase):
    @unittest.skipUnless(options.library == 'HtFLlib', 'disk-based framework')
    def test_disk_model_roundtrip(self):
        from flcore.trainmodel.models import BaseHeadSplit
        from flcore.clients.clientbase import save_item, load_item
        args = configure('Cifar10', 10)
        model = BaseHeadSplit(args, 0).eval()
        x = torch.randn(2, 3, 32, 32)
        with tempfile.TemporaryDirectory() as directory:
            save_item(model, 'Client_0', 'model', directory)
            restored = load_item('Client_0', 'model', directory)
            torch.testing.assert_close(restored(x), model(x))

    def test_dataset_input_dimensions(self):
        if options.library == 'HtFLlib':
            from flcore.trainmodel.models import BaseHeadSplit
            for dataset, classes, size, dim in [('Cifar10', 10, 32, 1600), ('Cifar100', 100, 32, 1600), ('TinyImagenet', 200, 64, 10816)]:
                with self.subTest(dataset=dataset):
                    args = configure(dataset, classes)
                    model = BaseHeadSplit(args, 0)
                    self.assertEqual(model.base.fc1[0].in_features, dim)
                    output = model(torch.randn(2, 3, size, size))
                    self.assertEqual(tuple(output.shape), (2, classes))
                    nn.CrossEntropyLoss()(output, torch.tensor([0, classes - 1])).backward()
                    for cid in (8, 9):
                        # Real full-width ViT construction, without allocating its weights.
                        with torch.device('meta'):
                            vit = BaseHeadSplit(args, cid)
                        self.assertEqual(vit.base.image_size, size)
                        self.assertIsInstance(vit.head, nn.Sequential)
                        self.assertIsInstance(vit.head[1], nn.Tanh)
                        self.assertEqual((vit.head[0].in_features, vit.head[0].out_features), (args.feature_dim, classes))
                        self.assertEqual((vit.head[2].in_features, vit.head[2].out_features), (classes, classes))
                        del vit
                    del model
        else:
            for dataset, classes, size, name in [('Cifar10', 10, 32, 'cnn'), ('Cifar100', 100, 32, 'cnn'), ('TinyImagenet', 200, 64, 'resnet18')]:
                with self.subTest(dataset=dataset):
                    args = configure(dataset, classes, ['-m', name])
                    output = args.model(torch.randn(2, 3, size, size))
                    self.assertEqual(tuple(output.shape), (2, classes))
                    nn.CrossEntropyLoss()(output, torch.tensor([0, classes - 1])).backward()
                    del args, output
                    gc.collect()

    @unittest.skipUnless(options.library == 'HtFLlib', 'heterogeneous heads')
    def test_two_layer_vit_head_and_feature_dim(self):
        from flcore.trainmodel.models import BaseHeadSplit
        from torchvision.models.vision_transformer import VisionTransformer
        for classes in (10, 100, 200):
            args = configure('Cifar10', classes, ['-fd', '128'])
            cnn = BaseHeadSplit(args, 0)
            self.assertEqual(tuple(cnn(torch.randn(2, 3, 32, 32)).shape), (2, classes))
            for factory in ('vit_b_16', 'vit_b_32'):
                cid = 8 if factory == 'vit_b_16' else 9
                def small_vit(image_size, num_classes):
                    return VisionTransformer(image_size=image_size, patch_size=16,
                        num_layers=1, num_heads=2, hidden_dim=32, mlp_dim=64,
                        num_classes=num_classes)
                with patch('torchvision.models.' + factory, small_vit):
                    model = BaseHeadSplit(args, cid)
                representation = torch.randn(2, 128)
                self.assertEqual(tuple(model.head(representation).shape), (2, classes))
                nn.CrossEntropyLoss()(model(torch.randn(2, 3, 32, 32)), torch.tensor([0, classes - 1])).backward()


class StateTests(unittest.TestCase):
    @unittest.skipUnless(options.library == 'HtFLlib', 'disk-based heterogeneous framework')
    def test_fedgh_broadcast_preserves_local_vit_second_layer(self):
        from flcore.clients.clientre import clientre
        from flcore.clients.clientbase import save_item, load_item
        from flcore.trainmodel.models import BaseHeadSplit
        from torchvision.models.vision_transformer import VisionTransformer
        from types import SimpleNamespace

        def small_vit(image_size, num_classes):
            return VisionTransformer(image_size=image_size, patch_size=16,
                num_layers=1, num_heads=2, hidden_dim=32, mlp_dim=64,
                num_classes=num_classes)

        for dataset, classes, size in [('Cifar10', 10, 32), ('Cifar100', 100, 32), ('TinyImagenet', 200, 64)]:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                args = configure(dataset, classes)
                shared = BaseHeadSplit(args, 0).head
                save_item(shared, 'Server', 'head', directory)
                with patch('torchvision.models.vit_b_16', small_vit):
                    model = BaseHeadSplit(args, 8)
                local_before = copy.deepcopy(model.head[2].state_dict())
                save_item(model, 'Client_8', 'model', directory)
                clientre.set_parameters(SimpleNamespace(role='Client_8', save_folder_name=directory))
                restored = load_item('Client_8', 'model', directory)
                for key, value in shared.state_dict().items():
                    torch.testing.assert_close(restored.head[0].state_dict()[key], value)
                for key, value in local_before.items():
                    torch.testing.assert_close(restored.head[2].state_dict()[key], value)
                self.assertIsInstance(restored.head[1], nn.Tanh)
                self.assertEqual(restored.head[2].in_features, classes)
                output = restored(torch.randn(2, 3, size, size))
                self.assertEqual(tuple(output.shape), (2, classes))
                optimizer = torch.optim.SGD(restored.parameters(), lr=0.01)
                optimizer.zero_grad()
                nn.CrossEntropyLoss()(output, torch.tensor([0, classes - 1])).backward()
                optimizer.step()
                self.assertTrue(any(not torch.equal(value, restored.head[2].state_dict()[key])
                                    for key, value in local_before.items()))

    @unittest.skipUnless(options.library == 'PFLlib', 'in-memory homogeneous framework')
    def test_server_independence_scheduler_and_broadcast(self):
        from flcore.servers.serverre import FedRE
        args = configure('Cifar10', 10, ['-nc', '2', '-lbs', '2', '-ld', 'True', '-ldg', '0.8', '-lr', '0.02', '-slr', '0.123'])
        data = [(torch.randn(3, 32, 32), torch.tensor(i % 2)) for i in range(4)]
        with patch('flcore.servers.serverbase.read_client_data', return_value=data), patch('flcore.clients.clientbase.read_client_data', return_value=data):
            server = FedRE(args, 0)
            self.assertIsNot(server.head, server.clients[0].model.head)
            self.assertAlmostEqual(server.opt_h.param_groups[0]['lr'], 0.123)
            before = copy.deepcopy(server.head.state_dict())
            server.clients[0].train()
            for key, value in before.items():
                torch.testing.assert_close(server.head.state_dict()[key], value)
            self.assertAlmostEqual(server.clients[0].optimizer.param_groups[0]['lr'], 0.016)
            server.send_models()
            for client in server.clients:
                for key, value in server.head.state_dict().items():
                    torch.testing.assert_close(client.model.head.state_dict()[key], value)



class CliTests(unittest.TestCase):
    def test_explicit_true_and_false(self):
        for flag in ('-ld', '-rjr', '-ab', '-ts'):
            self.assertFalse(getattr(parse_args([flag, 'False']), {'-ld': 'learning_rate_decay', '-rjr': 'random_join_ratio', '-ab': 'auto_break', '-ts': 'time_select'}[flag]))
            self.assertTrue(getattr(parse_args([flag, 'True']), {'-ld': 'learning_rate_decay', '-rjr': 'random_join_ratio', '-ab': 'auto_break', '-ts': 'time_select'}[flag]))


class DataTests(unittest.TestCase):
    def test_tinyimagenet_uses_only_original_training_pool(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'TinyImagenet'
            raw = target / 'rawdata/tiny-imagenet-200'
            for cid in range(3):
                folder = raw / 'train' / f'n{cid:08d}' / 'images'
                folder.mkdir(parents=True)
                for i in range(8):
                    Image.new('RGB', (64, 64), (cid * 60, i, 0)).save(folder / f'{i}.JPEG')
            validation = raw / 'val/images'
            validation.mkdir(parents=True)
            for i in range(6):
                Image.new('RGB', (64, 64)).save(validation / f'val_{i}.JPEG')
            with contextlib.redirect_stdout(io.StringIO()):
                generator.set_random_seed(0)
                generator.generate_dataset(str(target) + '/', 2, False, True, None)
            labels = []
            for subset in ('train', 'test'):
                for p in (target / subset).glob('*.npz'):
                    labels.extend(np.load(p, allow_pickle=True)['data'].item()['y'])
            self.assertEqual(len(labels), 24)
            self.assertEqual(np.bincount(labels).tolist(), [8, 8, 8])
            self.assertEqual(sum(len(np.load(p, allow_pickle=True)['data'].item()['y']) for p in (target / 'train').glob('*.npz')), 18)


if __name__ == '__main__':
    suite_class = {'models': ModelTests, 'state': StateTests, 'data': DataTests, 'cli': CliTests}[options.suite]
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(suite_class)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
