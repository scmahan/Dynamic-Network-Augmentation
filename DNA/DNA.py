import torch
import torch.nn as nn
import torchvision.transforms as T
import augmentations as A
from networks import get_model


class DifferentiableAugment(nn.Module):
    def __init__(self, sub_policy):
        super(DifferentiableAugment, self).__init__()
        self.sub_policy = sub_policy

    def forward(self, origin_images, probability, magnitude):
        images = origin_images
        adds = 0
        for i in range(len(self.sub_policy)):
            if probability[i].item() != 0.0:
                images = images - magnitude[i]
                adds = adds + magnitude[i]
        images = images.detach() + adds
        return images


class MixedAugment(nn.Module):
    def __init__(self, sub_policies):
        super(MixedAugment, self).__init__()
        self.sub_policies = sub_policies
        self._compile(sub_policies)

    def _compile(self, sub_policies):
        self._ops = nn.ModuleList()
        self._nums = len(sub_policies)
        for sub_policy in sub_policies:
            ops = DifferentiableAugment(sub_policy)
            self._ops.append(ops)

    def forward(self, origin_images, probabilities, magnitudes, weights, indices):
        bs = len(indices)
        I = torch.zeros([bs,bs,bs])
        for i in range(bs):
            I[i][i][i] = 1.
        if torch.cuda.is_available():
            I = I.cuda()
        return sum(torch.tensordot(I[i], sum((w * op(origin_images, p, m) if indices[i].item() == j else w
                   for j, (p, m, w, op) in
                   enumerate(zip(probabilities[i], magnitudes[i], weights[i], self._ops)))), 1)
                   for i in range(bs))
    
    
class AugmentImages(nn.Module):
    def __init__(self, sub_policies):
        super(AugmentImages, self).__init__()
        self.sub_policies = sub_policies
        self.ToP = T.ToPILImage()
        self.ToT = T.ToTensor()

    def forward(self, origin_images, probabilities, magnitudes, indices):
        bs = len(indices)
        k = probabilities.shape[2]
        images = torch.zeros_like(origin_images)
        if torch.cuda.is_available():
            images = images.cuda()
        for i in range(bs):
            image = self.ToP(origin_images[i])
            for l in range(k):
                if probabilities[i][indices[i].item()][l] != 0:
                    f = self.sub_policies[indices[i]][l]
                    image = A.apply_augment(image, f, magnitudes[i][indices[i].item()][l])
            image = self.ToT(image)
            images[i] = image
        return images
    

class DNA(nn.Module):
    def __init__(self, model_name, sub_policies, temperature=0.5, num_targets=10):
        super().__init__()
        self.sub_policies = sub_policies
        self.temperature = torch.tensor(temperature)
        self.num_targets = num_targets
        
        num_sub_policies = len(sub_policies)
        num_ops = len(sub_policies[0])
        self.augnet = get_model(model_name, num_targets, policy_shape=(num_sub_policies,num_ops))
        self.classnet = get_model(model_name, num_targets)
        self.mix_augment = MixedAugment(sub_policies)
        self.augment_img = AugmentImages(sub_policies)
        self.mode = "search"

    def set_mode(self, value):
        assert value in ["search", "train", "test"]
        self.mode = value
        if value == "train":
            for param in self.augnet.parameters():
                param.requires_grad = False
    
    def forward_augnet(self, x):
        p, m, pi = self.augnet(x)
        #print(p[0])
        self.probabilities = p
        self.magnitudes = m
        self.ops_weights = pi        
    
    def sample(self):
        probabilities_dist = torch.distributions.RelaxedBernoulli(
            self.temperature, self.probabilities)
        sample_probabilities = probabilities_dist.rsample()
        sample_probabilities = sample_probabilities.clamp(0.0, 1.0)
        self.sample_probabilities_index = sample_probabilities >= 0.5
        self.sample_probabilities = \
            self.sample_probabilities_index.float() - sample_probabilities.detach() + sample_probabilities

        ops_weights_dist = torch.distributions.RelaxedOneHotCategorical(
            self.temperature, logits=self.ops_weights)
        sample_ops_weights = ops_weights_dist.rsample()
        sample_ops_weights = sample_ops_weights.clamp(0.0, 1.0)
        self.sample_ops_weights_index = torch.max(sample_ops_weights, dim=-1, keepdim=True)[1]
        one_h = torch.zeros_like(sample_ops_weights).scatter_(-1, self.sample_ops_weights_index, 1.0)
        self.sample_ops_weights = one_h - sample_ops_weights.detach() + sample_ops_weights
    
    def forward_search(self, x):
        self.forward_augnet(x)
        self.sample()
        x_aug = self.mix_augment(
            x, self.sample_probabilities, self.magnitudes, self.sample_ops_weights, self.sample_ops_weights_index)
        output = self.classnet(x_aug)
        return output    
    
    def forward_train(self, x):
        self.forward_augnet(x)
        self.sample()
        x_aug = self.augment_img(
            x, self.sample_probabilities, self.magnitudes, self.sample_ops_weights_index)
        output = self.classnet(x_aug)
        return output

    def forward_test(self, x):
        return self.classnet(x)

    def forward(self, x):
        if self.mode == "search":
            return self.forward_search(x)
        elif self.mode == "train":
            return self.forward_train(x)
        else:
            return self.forward_test(x)

    