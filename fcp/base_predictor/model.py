import numpy as np
from sklearn.linear_model import LinearRegression
from fcp.base_predictor.data import build_strided_feature
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import copy


class BasePredictor():

    def __init__(self,):
        ...

    def fit(self,):

        ...

    def generate_residual_dataset(self,):

        ...


class LOOBootstrapPredictor(BasePredictor):
    """
    train point predictor and obtain residuals using the given data
    """

    def __init__(self, num_estimators, stride):
        self.num_estimators = num_estimators
        self.stride = stride

    def fit(self, x, y, train_test_ratio):

        n_train = int(x.shape[0]*(train_test_ratio[0]/(np.sum(train_test_ratio))))

        self.x_train = x[:n_train]
        self.x_test = x[n_train:]
        self.y_train = y[:n_train]
        self.y_test = y[n_train:]

        dim_y = self.y_train.shape[1]
        n = self.x_train.shape[0]
        n1 = self.x_test.shape[0]
        N = n-self.stride+1  # Total training data each one-step predictor sees

        full_resid = np.ones((n+n1,dim_y))*np.inf
        train_pred_interval_center = np.ones((n,dim_y))*np.inf
        test_pred_interval_center = np.ones((n1,dim_y))*np.inf

        # We make prediction every s step ahead, so these are feature the model sees
        train_pred_idx = np.arange(0, n, self.stride)
        test_pred_idx = np.arange(n, n+n1, self.stride)

        # Only contains features that are observed every stride steps
        x_full = np.vstack([self.x_train[train_pred_idx], self.x_test[test_pred_idx-n]])
        n_sub, n1_sub = len(train_pred_idx), len(test_pred_idx)

        for s in range(self.stride):
            ''' 
            Create containers for predictions 
            '''

            # hold indices of training data for each f^b
            bootstrap_samples_idx = self.generate_bootstrap_samples(N, N, self.num_estimators)
            # for i-th column, it shows which f^b uses i in training (so exclude in aggregation)
            in_boot_sample = np.zeros((self.num_estimators, N), dtype=bool)
            # hold predictions from each f^b for fX and sigma&b for sigma
            boot_predictionsFX = np.zeros((self.num_estimators, n_sub+n1_sub, dim_y))
            # We actually would just use n1sub rows, as we only observe this number of features
            out_sample_predictFX = np.zeros((n, n1_sub, dim_y))

            ''' 
            Start bootstrap prediction 
            '''
            for b in range(self.num_estimators):
                x_boot, y_boot = self.x_train[bootstrap_samples_idx[b],:], self.y_train[s:s+N][bootstrap_samples_idx[b],:]
                in_boot_sample[b, bootstrap_samples_idx[b]] = True
                boot_fX_pred = self.one_boot_prediction(x_boot, y_boot, x_full)
                boot_predictionsFX[b] = boot_fX_pred

            ''' 
            Obtain LOO residuals (train and test) and prediction for test data 
            '''
            # Consider LOO, but here ONLY for the indices being predicted
            for j, i in enumerate(train_pred_idx):
                # j: counter
                # i: actual index X_{0+j*stride}

                if i < N:
                    b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
                    if len(b_keep) == 0:
                        # All bootstrap estimators are trained on this model
                        b_keep = 0  # More rigorously, it should be None, but in practice, the difference is minor
                else:
                    # This feature is not used in training, but used in prediction
                    b_keep = range(self.num_estimators)

                pred_iFX = boot_predictionsFX[b_keep, j].mean(axis=0)
                pred_testFX = boot_predictionsFX[b_keep, n_sub:].mean(axis=0)

                # Populate the training prediction
                # We add s because of multi-step procedure, so f(X_t) is for Y_t+s
                true_idx = min(i+s, n-1)
                train_pred_interval_center[true_idx] = pred_iFX
                resid_LOO = self.y_train[true_idx] - pred_iFX
                out_sample_predictFX[i] = pred_testFX
                full_resid[true_idx] = resid_LOO

            sorted_out_sample_predictFX = out_sample_predictFX[train_pred_idx].mean(0)  # length ceil(n1/stride)
            pred_idx = np.minimum(test_pred_idx-n+s, n1-1)
            test_pred_interval_center[pred_idx] = sorted_out_sample_predictFX
            pred_full_idx = np.minimum(test_pred_idx+s, n+n1-1)
            resid_out_sample = self.y_test[pred_idx] - sorted_out_sample_predictFX
            full_resid[pred_full_idx] = resid_out_sample

        # Sanity check
        if (full_resid == np.inf).sum() > 0:
            raise ValueError('some residuals were not computed')
        
        self.full_resid = full_resid
        self.train_resid = full_resid[:n]
        self.test_resid = full_resid[n:]
        self.train_pred = train_pred_interval_center
        self.test_pred = test_pred_interval_center

    @staticmethod
    def one_boot_prediction(x_train, y_train, x_test):

        estimator = LinearRegression()
        estimator.fit(x_train,y_train)

        return estimator.predict(x_test)
    
    @staticmethod
    def generate_bootstrap_samples(n, m, B):
        '''
        Returns
        -------
            B-by-m matrix, where row b gives the indices for b-th bootstrap sample
        '''
        samples_idx = np.zeros((B, m), dtype=int)

        for b in range(B):
            sample_idx = np.random.choice(n, m)
            samples_idx[b, :] = sample_idx

        return samples_idx
    
    def generate_residual_dataset(self, valid_test_ratio=None):

        if valid_test_ratio:
            n = int(len(self.test_resid) * valid_test_ratio[0] / (valid_test_ratio[0]+valid_test_ratio[1]))

            x_valid = self.x_test[:n]
            x_test_ = self.x_test[n:]
            x_full = np.concatenate([self.x_train, self.x_test], axis=0)

            resid_valid = self.test_resid[:n]
            resid_test = self.test_resid[n:]
            resid_full = np.concatenate([self.train_resid, self.test_resid], axis=0)

            pred_valid = self.test_pred[:n]
            pred_test = self.test_pred[n:]
            pred_full = np.concatenate([self.train_pred, self.test_pred], axis=0)

            y_valid = self.y_test[:n]
            y_test = self.y_test[n:]

            return {"x_train" : self.x_train, 
                    "x_valid" : x_valid, 
                    "x_test" : x_test_,
                    "x_full" : x_full,
                    "resid_train" : self.train_resid,
                    "resid_valid" : resid_valid,
                    "resid_test" : resid_test,
                    "resid_full" : resid_full,
                    "pred_train" : self.train_pred,
                    "pred_valid" : pred_valid,
                    "pred_test" : pred_test,
                    "pred_full" : pred_full,
                    "y_train" : self.y_train,
                    "y_valid" : y_valid,
                    "y_test" : y_test}

        else:
            x_full = np.concatenate([self.x_train, self.x_test], axis=0)
            resid_full = np.concatenate([self.train_resid, self.test_resid], axis=0)
            pred_full = np.concatenate([self.train_pred, self.test_pred], axis=0)
            return {"x_train" : self.x_train, 
                    "x_test" : self.x_test, 
                    "x_full" : x_full,
                    "resid_train" : self.train_resid,
                    "resid_test" : self.test_resid,
                    "resid_full" : resid_full,
                    "pred_train" : self.train_pred,
                    "pred_test" : self.test_pred,
                    "pred_full" : pred_full}

        
class LSTMPredictor(BasePredictor):

    def __init__(self, dim_model, context_window):
        self.dim_model = dim_model
        self.context_window = context_window

    def fit(self, x, y, predictor_data_ratio, train_valid_ratio,  
            max_epoch, batch_size, learning_rate, early_stop, device):
        
        n = int(x.shape[0]*predictor_data_ratio)
        
        x_predictor = x[:n]
        y_predictor = y[:n]

        n_train = int(x_predictor.shape[0]*train_valid_ratio[0])

        x_train = x_predictor[:n_train]
        y_train = y_predictor[:n_train]
        x_valid = x_predictor[x_train.shape[0]-self.context_window:,:]
        y_valid = y_predictor[x_train.shape[0]-self.context_window:,:]
        x_test = x[n-self.context_window:,:]
        y_test = y[n-self.context_window:,:]

        x_validtest = x[x_train.shape[0]:,:]
        y_validtest = y[x_train.shape[0]:,:]
        
        x_train_strided = build_strided_feature(x_train, self.context_window)
        x_valid_strided = build_strided_feature(x_valid, self.context_window)
        x_test_strided = build_strided_feature(x_test, self.context_window)
        y_train_strided = build_strided_feature(y_train, self.context_window)
        y_valid_strided = build_strided_feature(y_valid, self.context_window)
        y_test_strided = build_strided_feature(y_test, self.context_window)

        train_dataset = TensorDataset(torch.tensor(x_train_strided, dtype=torch.float32), 
                                      torch.tensor(y_train_strided, dtype=torch.float32))
        valid_dataset = TensorDataset(torch.tensor(x_valid_strided, dtype=torch.float32), 
                                      torch.tensor(y_valid_strided, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(x_test_strided, dtype=torch.float32), 
                                      torch.tensor(y_test_strided, dtype=torch.float32))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        # initialize model
        lstm_model = LSTM(x_train.shape[-1], self.dim_model, y_train.shape[-1])
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
        best_val_loss = np.inf

        # training model loop
        lstm_model.to(device)
        for epoch in range(max_epoch):

            lstm_model.train()
            total_loss = 0
            for x_batch, y_batch in tqdm(train_dataloader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                y_pred = lstm_model(x_batch)

                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}, training loss: {avg_loss:.4f}")

            # Validation
            lstm_model.eval()
            total_loss = 0.
            with torch.no_grad():
                for x_val, y_val in tqdm(valid_dataloader):
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    y_pred_val = lstm_model(x_val)
                    loss = criterion(y_pred_val, y_val)
                    total_loss += loss.item()

            avg_val_loss = total_loss / len(valid_dataloader)
            print(f"Epoch {epoch + 1} validation loss: {avg_val_loss:.4f}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = copy.deepcopy(lstm_model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= early_stop:
                    break

        lstm_model.load_state_dict(best_model)
        lstm_model.eval()

        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        pred_valid = []
        resid_valid = []
        with torch.no_grad():
            for x, y in tqdm(valid_dataloader):
                x = x.to(device)
                y = y.to(device)

                pred = lstm_model(x) # (1, context_window, dim_outcome)
                resid = y - pred

                pred_valid.append(pred[0,-1,:].cpu().numpy())
                resid_valid.append(resid[0,-1,:].cpu().numpy())

        pred_test = []
        resid_test = []
        with torch.no_grad():
            for x, y in tqdm(test_dataloader):
                x = x.to(device)
                y = y.to(device)

                pred = lstm_model(x) # (1, context_window, dim_outcome)
                resid = y - pred

                pred_test.append(pred[0,-1,:].cpu().numpy())
                resid_test.append(resid[0,-1,:].cpu().numpy())

        self.x_train = x_train
        self.x_valid = x_valid
        self.x_test = x_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test
        self.x_validtest = x_validtest
        self.y_validtest = y_validtest

        self.resid_valid = np.array(resid_valid)
        self.resid_test = np.array(resid_test)
        self.resid_validtest = np.concatenate([resid_valid,resid_test], axis=0)
        
        self.pred_valid = np.array(pred_valid)
        self.pred_test = np.array(pred_test)
        self.pred_validtest = np.concatenate([pred_valid,pred_test], axis=0)

    def generate_residual_dataset(self, train_valid_test_ratio=None):

        if len(train_valid_test_ratio) == 2:
            """
            split dataset into train and test set
            """
            train_ratio = train_valid_test_ratio[0]/(train_valid_test_ratio[0]+train_valid_test_ratio[1])
            n_train = int((self.pred_validtest.shape[0])*train_ratio)

            return {"x_train" : self.x_validtest[:n_train,:], 
                    "x_test" : self.x_validtest[n_train:,:],
                    "x_full" : self.x_validtest,
                    "resid_train" : self.resid_validtest[:n_train,:],
                    "resid_test" : self.resid_validtest[n_train:,:],
                    "resid_full" : self.resid_validtest,
                    "pred_train" : self.pred_validtest[:n_train,:],
                    "pred_test" : self.pred_validtest[n_train:],
                    "pred_full" : self.pred_validtest,
                    "y_train" : self.y_validtest[:n_train,:], 
                    "y_test" : self.y_validtest[n_train:,:],
                    "y_full" : self.y_validtest,
                    }

        elif len(train_valid_test_ratio) == 3:
            """
            split dataset into train, validation, and test set
            """
            train_ratio = train_valid_test_ratio[0]/np.sum(train_valid_test_ratio)
            trainvalid_ratio = (train_valid_test_ratio[0]+train_valid_test_ratio[1])/np.sum(train_valid_test_ratio)
            n_train = int((self.pred_validtest.shape[0])*train_ratio)
            n_valid = int((self.pred_validtest.shape[0])*trainvalid_ratio)

            return {"x_train" : self.x_validtest[:n_train,:], 
                    "x_valid" : self.x_validtest[n_train:n_valid,:],
                    "x_test" : self.x_validtest[n_valid:,:],
                    "x_full" : self.x_validtest,
                    "resid_train" : self.resid_validtest[:n_train,:],
                    "resid_valid" : self.resid_validtest[n_train:n_valid,:],
                    "resid_test" : self.resid_validtest[n_valid:,:],
                    "resid_full" : self.resid_validtest,
                    "pred_train" : self.pred_validtest[:n_train,:],
                    "pred_valid" : self.pred_validtest[n_train:n_valid,:],
                    "pred_test" : self.pred_validtest[n_valid:,:],
                    "pred_full" : self.pred_validtest,
                    "y_train" : self.y_validtest[:n_train,:], 
                    "y_valid" : self.y_validtest[n_train:n_valid,:],
                    "y_test" : self.y_validtest[n_valid:,:],
                    "y_full" : self.y_validtest}

class LSTM(torch.nn.Module):

    def __init__(self, input_dim, model_dim, outcome_dim):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, model_dim, batch_first=True)
        self.linear = torch.nn.Linear(model_dim, outcome_dim)

    def forward(self, x):
        """
        args
        ----
            x: sequential input, (batch_size, past_window, input_dim)

        returns
        -------
            output: hidden states, (batch_size, past_window, model_dim)
        """

        output, (h, c) = self.lstm(x)

        return self.linear(output) # no activation