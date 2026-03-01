"""
Payment Gateway Integration - VNPay & MoMo
Handles payment creation, verification, and callbacks
"""

import hashlib
import hmac
import urllib.parse
import requests
import json
import time
from datetime import datetime
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VNPayPayment:
    """VNPay Payment Gateway Integration"""
    
    def __init__(self, tmn_code: str, secret_key: str, payment_url: str, return_url: str):
        """
        Initialize VNPay payment gateway
        
        Args:
            tmn_code: VNPay Terminal Code (Mã website)
            secret_key: VNPay Secret Key (Mã bảo mật)
            payment_url: VNPay Payment URL
            return_url: Return URL after payment
        """
        self.tmn_code = tmn_code
        self.secret_key = secret_key
        self.payment_url = payment_url
        self.return_url = return_url
    
    def create_payment_url(self, 
                          order_id: str, 
                          amount: int,
                          order_desc: str,
                          order_type: str = 'other',
                          locale: str = 'vn',
                          ip_addr: str = '127.0.0.1') -> str:
        """
        Create VNPay payment URL
        
        Args:
            order_id: Unique order ID
            amount: Amount in VND (will be multiplied by 100)
            order_desc: Order description
            order_type: Order type (billpayment, other, etc.)
            locale: Language (vn or en)
            ip_addr: Customer IP address
            
        Returns:
            Payment URL for redirect
        """
        try:
            # VNPay requires amount in smallest unit (xu)
            vnp_amount = int(amount * 100)
            
            # Create request data
            vnp_params = {
                'vnp_Version': '2.1.0',
                'vnp_Command': 'pay',
                'vnp_TmnCode': self.tmn_code,
                'vnp_Amount': str(vnp_amount),
                'vnp_CurrCode': 'VND',
                'vnp_TxnRef': order_id,
                'vnp_OrderInfo': order_desc,
                'vnp_OrderType': order_type,
                'vnp_Locale': locale,
                'vnp_ReturnUrl': self.return_url,
                'vnp_IpAddr': ip_addr,
                'vnp_CreateDate': datetime.now().strftime('%Y%m%d%H%M%S')
            }
            
            # Sort parameters
            sorted_params = sorted(vnp_params.items())
            
            # Create query string
            query_string = '&'.join([f"{key}={urllib.parse.quote_plus(str(val))}" 
                                    for key, val in sorted_params])
            
            # Create secure hash
            secure_hash = self._create_signature(query_string)
            
            # Build final URL
            payment_url = f"{self.payment_url}?{query_string}&vnp_SecureHash={secure_hash}"
            
            logger.info(f"Created VNPay payment URL for order {order_id}")
            return payment_url
            
        except Exception as e:
            logger.error(f"Error creating VNPay payment URL: {e}")
            raise
    
    def verify_payment(self, params: Dict[str, str]) -> Tuple[bool, str, Dict]:
        """
        Verify VNPay payment callback
        
        Args:
            params: Callback parameters from VNPay
            
        Returns:
            (is_valid, message, payment_data)
        """
        try:
            # Extract secure hash
            vnp_secure_hash = params.pop('vnp_SecureHash', '')
            
            # Sort and create query string
            sorted_params = sorted(params.items())
            query_string = '&'.join([f"{key}={urllib.parse.quote_plus(str(val))}" 
                                    for key, val in sorted_params])
            
            # Verify signature
            expected_hash = self._create_signature(query_string)
            
            if vnp_secure_hash != expected_hash:
                logger.warning(f"Invalid VNPay signature for order {params.get('vnp_TxnRef')}")
                return False, "Invalid signature", {}
            
            # Check response code
            response_code = params.get('vnp_ResponseCode', '')
            transaction_status = params.get('vnp_TransactionStatus', '')
            
            if response_code == '00' and transaction_status == '00':
                payment_data = {
                    'order_id': params.get('vnp_TxnRef'),
                    'amount': int(params.get('vnp_Amount', 0)) / 100,  # Convert back to VND
                    'bank_code': params.get('vnp_BankCode'),
                    'bank_tran_no': params.get('vnp_BankTranNo'),
                    'card_type': params.get('vnp_CardType'),
                    'pay_date': params.get('vnp_PayDate'),
                    'transaction_no': params.get('vnp_TransactionNo'),
                    'status': 'success'
                }
                logger.info(f"VNPay payment successful for order {payment_data['order_id']}")
                return True, "Payment successful", payment_data
            else:
                logger.warning(f"VNPay payment failed: code={response_code}, status={transaction_status}")
                return False, f"Payment failed: {response_code}", {}
                
        except Exception as e:
            logger.error(f"Error verifying VNPay payment: {e}")
            return False, str(e), {}
    
    def _create_signature(self, data: str) -> str:
        """Create HMAC SHA512 signature"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()


class MoMoPayment:
    """MoMo Payment Gateway Integration"""
    
    def __init__(self, partner_code: str, access_key: str, secret_key: str, 
                 payment_url: str, return_url: str, notify_url: str):
        """
        Initialize MoMo payment gateway
        
        Args:
            partner_code: MoMo Partner Code
            access_key: MoMo Access Key
            secret_key: MoMo Secret Key
            payment_url: MoMo API endpoint
            return_url: Return URL after payment
            notify_url: IPN callback URL
        """
        self.partner_code = partner_code
        self.access_key = access_key
        self.secret_key = secret_key
        self.payment_url = payment_url
        self.return_url = return_url
        self.notify_url = notify_url
    
    def create_payment_url(self,
                          order_id: str,
                          amount: int,
                          order_info: str,
                          request_type: str = 'captureWallet',
                          extra_data: str = '') -> Tuple[bool, str, Dict]:
        """
        Create MoMo payment URL
        
        Args:
            order_id: Unique order ID
            amount: Amount in VND
            order_info: Order information
            request_type: Payment type (captureWallet, payWithATM, etc.)
            extra_data: Additional data (optional)
            
        Returns:
            (success, payment_url_or_error, response_data)
        """
        try:
            request_id = f"{order_id}_{int(time.time())}"
            
            # Create request data
            raw_data = (f"accessKey={self.access_key}"
                       f"&amount={amount}"
                       f"&extraData={extra_data}"
                       f"&ipnUrl={self.notify_url}"
                       f"&orderId={order_id}"
                       f"&orderInfo={order_info}"
                       f"&partnerCode={self.partner_code}"
                       f"&redirectUrl={self.return_url}"
                       f"&requestId={request_id}"
                       f"&requestType={request_type}")
            
            # Create signature
            signature = self._create_signature(raw_data)
            
            # Request payload
            payload = {
                'partnerCode': self.partner_code,
                'accessKey': self.access_key,
                'requestId': request_id,
                'amount': str(amount),
                'orderId': order_id,
                'orderInfo': order_info,
                'redirectUrl': self.return_url,
                'ipnUrl': self.notify_url,
                'extraData': extra_data,
                'requestType': request_type,
                'signature': signature,
                'lang': 'vi'
            }
            
            # Send request to MoMo
            response = requests.post(
                self.payment_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            result = response.json()
            
            if result.get('resultCode') == 0:
                pay_url = result.get('payUrl', '')
                logger.info(f"Created MoMo payment URL for order {order_id}")
                return True, pay_url, result
            else:
                error_msg = result.get('message', 'Unknown error')
                logger.error(f"MoMo payment creation failed: {error_msg}")
                return False, error_msg, result
                
        except Exception as e:
            logger.error(f"Error creating MoMo payment: {e}")
            return False, str(e), {}
    
    def verify_payment(self, params: Dict) -> Tuple[bool, str, Dict]:
        """
        Verify MoMo payment callback
        
        Args:
            params: Callback parameters from MoMo
            
        Returns:
            (is_valid, message, payment_data)
        """
        try:
            # Extract signature
            received_signature = params.get('signature', '')
            
            # Create raw signature data
            raw_data = (f"accessKey={self.access_key}"
                       f"&amount={params.get('amount', '')}"
                       f"&extraData={params.get('extraData', '')}"
                       f"&message={params.get('message', '')}"
                       f"&orderId={params.get('orderId', '')}"
                       f"&orderInfo={params.get('orderInfo', '')}"
                       f"&orderType={params.get('orderType', '')}"
                       f"&partnerCode={self.partner_code}"
                       f"&payType={params.get('payType', '')}"
                       f"&requestId={params.get('requestId', '')}"
                       f"&responseTime={params.get('responseTime', '')}"
                       f"&resultCode={params.get('resultCode', '')}"
                       f"&transId={params.get('transId', '')}")
            
            # Verify signature
            expected_signature = self._create_signature(raw_data)
            
            if received_signature != expected_signature:
                logger.warning(f"Invalid MoMo signature for order {params.get('orderId')}")
                return False, "Invalid signature", {}
            
            # Check result code
            result_code = params.get('resultCode', -1)
            
            if result_code == 0:
                payment_data = {
                    'order_id': params.get('orderId'),
                    'amount': int(params.get('amount', 0)),
                    'transaction_id': params.get('transId'),
                    'pay_type': params.get('payType'),
                    'response_time': params.get('responseTime'),
                    'status': 'success'
                }
                logger.info(f"MoMo payment successful for order {payment_data['order_id']}")
                return True, "Payment successful", payment_data
            else:
                message = params.get('message', 'Payment failed')
                logger.warning(f"MoMo payment failed: code={result_code}, message={message}")
                return False, message, {}
                
        except Exception as e:
            logger.error(f"Error verifying MoMo payment: {e}")
            return False, str(e), {}
    
    def _create_signature(self, data: str) -> str:
        """Create HMAC SHA256 signature"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()


def get_client_ip(request) -> str:
    """Get client IP address from request"""
    x_forwarded_for = request.headers.get('X-Forwarded-For')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    return request.remote_addr or '127.0.0.1'
